import os
from typing import Tuple
import PIL
import tqdm

import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class SingleTaskDataset(torch.utils.data.Dataset):
    """
    Unified dataset class for single-task learning (classification or regression).
    Supports both Face256 and Daxing dataset structures with configurable directory depth.
    """
    def __init__(self, root, annotations_path, label='Sex', type='classification', dataset_type='train', 
                 transform=None, target_transform=None, loader=default_loader, args=None, 
                 directory_depth=2):
        """
        Args:
            root: Root directory of the dataset
            annotations_path: Path to the CSV annotation file
            label: Name of the target label column
            type: Task type, either 'classification' or 'regression'
            dataset_type: One of 'train', 'validation', 'test', or 'predict'
            transform: Image transformation pipeline
            target_transform: Target transformation pipeline
            loader: Image loader function
            args: Arguments object containing log_dir and output_dir
            directory_depth: Directory structure depth (1 for Daxing, 2 for Face256)
        """
        assert os.path.exists(root), f"{root} not exists"
        self.root = root
        self.annotations_path = annotations_path
        self.label = label
        self.type = type
        self.dataset_type = dataset_type
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.directory_depth = directory_depth

        self.log = open(os.path.join(args.log_dir, f'log_{dataset_type}set.txt'), 'w')
        self.output_dir = args.output_dir

        self.annotations, self.mean, self.std = self._load_annotations()
        self.samples = self._parse()

        self.log.close()

    def update_statistics(self, mean, std):
        """Update mean and standard deviation for normalization"""
        self.mean = mean
        self.std = std

    def get_statistics(self):
        """Get mean and standard deviation for normalization"""
        return self.mean, self.std

    def _load_annotations(self):
        """
        Load and parse annotations from CSV file.
        
        Purpose:
            - Read annotation CSV file with header line
            - Filter annotations by dataset_type (train/validation/test)
            - Extract target label for each sample (identified by 'eid')
            - Compute statistics (mean/std) for normalization
        
        CSV Format:
            - First line: header with column names
            - Following lines: one annotation per line, comma-separated
            - Required columns: 'eid' (entity/person ID), 'type' (dataset split)
            - Target column: specified by self.label parameter
        
        Returns:
            Tuple[List[Dict], float, float]:
                - annotations (List[Dict]): List of annotation dictionaries, each containing:
                    - 'eid': Entity/person identifier (str)
                    - <label>: Target value (float) for the specified label
                - mean (float): Mean value of the target label (0.0 for classification)
                - std (float): Standard deviation of the target label (1.0 for classification)
        
        Special Cases:
            - For dataset_type='predict': Returns (None, 0.0, 1.0)
            - For classification tasks: mean=0.0, std=1.0
            - For regression tasks: computed from non-NaN values
            - Samples with NaN target values are filtered out (except for 'predict' mode)
        
        Example annotation dict: {'eid': '12345', 'Height': 175.5}
        """
        if self.dataset_type == 'predict':
            return None, 0.0, 1.0
        
        # TODO: Implement annotation loading logic here
        # This method should be implemented according to specific dataset requirements
        pass
    
    def _parse(self):
        """
        Parse directory structure and match images with annotations.
        Supports both 1-level (Daxing) and 2-level (Face256) directory structures.
        """
        persons = dict()
        
        # Parse directory structure based on directory_depth
        if self.directory_depth == 2:
            # Face256 structure: root/dir/eid/
            for dir in os.listdir(self.root):
                dir_path = os.path.join(self.root, dir)
                if os.path.isdir(dir_path):
                    for subdir in os.listdir(dir_path):
                        path = os.path.join(dir_path, subdir)
                        if os.path.isdir(path):
                            if subdir not in persons:
                                persons[subdir] = path
                            else:
                                raise ValueError(f"{subdir} already exists")
        elif self.directory_depth == 1:
            # Daxing structure: root/eid/
            for subdir in os.listdir(self.root):
                path = os.path.join(self.root, subdir)
                if os.path.isdir(path):
                    if subdir not in persons:
                        persons[subdir] = path
                    else:
                        raise ValueError(f"{subdir} already exists")
        else:
            raise ValueError(f"Unsupported directory_depth: {self.directory_depth}")
        
        new_annotations = self.annotations

        # Parse annotations and match with image paths
        self.log.write('parsing annotations:' + '\n')
        samples = list()
        
        if new_annotations is None:
            # Predict mode: no annotations, just collect all images
            for subdir, path in persons.items():
                for img_name in os.listdir(path):
                    extension = os.path.splitext(img_name)[1].lower()
                    if extension in ['.jpg', '.jpeg', '.png']:
                        item = dict()
                        item['eid'] = subdir
                        item['path'] = os.path.join(path, img_name)
                        item[self.label] = np.nan
                        samples.append(item)
            self.log.write(f"number of samples: {len(samples)}\n")
        else:
            # Training/validation/test mode: match annotations with images
            for annotation_dict in tqdm.tqdm(new_annotations, desc='Parsing Dataset'):
                eid = annotation_dict['eid']
                if eid in persons:
                    path = persons[eid]
                    for img_name in os.listdir(path):
                        extension = os.path.splitext(img_name)[1].lower()
                        if extension in ['.jpg', '.jpeg', '.png']:
                            item = annotation_dict.copy()
                            item['path'] = os.path.join(path, img_name)
                            samples.append(item)
                else:
                    self.log.write(f"{eid} not exists\n")
            self.log.write(f"number of samples: {len(samples)}\n")

        return samples

    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Get a single sample from the dataset.
        
        Returns:
            Tuple[Tensor, str, str, Union[float, int]]:
                - img: Transformed image tensor
                - eid: Entity/person identifier
                - path: Image file path
                - target: Normalized target value
        """
        item = self.samples[index]
        eid = item['eid']
        path = item['path']
        img = self.loader(path)
        target = item[self.label]
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            # Normalization
            if self.type == 'regression':
                target = (target - self.mean) / self.std
            elif self.type == 'classification':
                target = target
            else:
                raise ValueError(f"{self.type} not supported")

        return img, eid, path, target


# Aliases for backward compatibility
Face256_Single = SingleTaskDataset
Daxing_Single = SingleTaskDataset


class LabelTransform(nn.Module):
    def __init__(self, labels):
        super(LabelTransform, self).__init__()
        self.labels = labels
        self.means = [0.0] * len(labels)
        self.stds = [1.0] * len(labels)
        # self.transform = v2.Normalize(self.means, self.stds)
        # self.totensor = v2.ToTensor()

    def update_statistics(self, means, stds):
        self.means = means
        self.stds = stds
        # self.transform = v2.Normalize(self.means, self.stds)

    def forward(self, target):
        # return self.transform(target)
        return (target - torch.tensor(self.means)) / torch.tensor(self.stds)
    
    def unnormalize(self, outputs):
        # outputs: Tensor [batch_size, num_labels]
        return outputs * torch.tensor(self.stds, device=outputs.device) + torch.tensor(self.means, device=outputs.device)


def build_dataset(is_train, args, **kwargs):
    """
    Build dataset based on args.dataset configuration.
    All datasets containing 'single' will use SingleTaskDataset with appropriate directory_depth.
    """
    transform = build_transform(is_train, args)
    dataset_type = kwargs['dataset_type'] if 'dataset_type' in kwargs else 'train' if is_train else 'validation'
    
    # Unified handling for all 'single' datasets
    if 'single' in args.dataset:
        # Determine directory depth based on dataset name
        directory_depth = 1
        
        # Create SingleTaskDataset with appropriate configuration
        dataset = SingleTaskDataset(
            root=args.data_path,
            annotations_path=args.annotation_path,
            label=args.label,
            type=args.type,
            dataset_type=dataset_type,
            transform=transform,
            target_transform=None,
            args=args,
            directory_depth=directory_depth
        )
    else:
        raise NotImplementedError(f"dataset {args.dataset} not supported")

    return dataset


def build_dataset_split(dataset_name, dataset_type, args, **kwargs):
    """
    Build dataset with explicit split paths.
    Unified handling for all datasets using SingleTaskDataset.
    """
    if 'train' in dataset_type:
        is_train = True
        data_path = args.train_data_path
        annotation_path = args.train_annotation_path
    elif 'val' in dataset_type:
        is_train = False
        data_path = args.val_data_path
        annotation_path = args.val_annotation_path
    elif 'test' in dataset_type:
        is_train = False
        data_path = args.test_data_path
        annotation_path = args.test_annotation_path
    else:
        raise ValueError(f"dataset_type {dataset_type} not supported")
    
    transform = build_transform(is_train, args)
    
    # Determine directory depth based on dataset name
    directory_depth = 1
    
    # Create SingleTaskDataset with appropriate configuration
    dataset = SingleTaskDataset(
        root=data_path,
        annotations_path=annotation_path,
        label=args.label,
        type=args.type,
        dataset_type=dataset_type,
        transform=transform,
        target_transform=None,
        args=args,
        directory_depth=directory_depth
    )
    
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        if args.aa == 'no_aug':
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
        else:
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
