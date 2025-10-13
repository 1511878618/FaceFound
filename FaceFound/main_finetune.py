import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import logging

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, ResNet18_Weights, vit_l_16, ViT_L_16_Weights
import pandas as pd

import timm
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from early_stopping_pytorch import EarlyStopping

import utils.lr_decay as lrd
import utils.misc as misc
from utils.datasets import build_dataset
from utils.pos_embed import interpolate_pos_embed
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import utils.models_pvt, utils.models_swin, utils.models_vit

from engine_finetune import evaluate, train_one_epoch, predict


def get_args_parser():
    parser = argparse.ArgumentParser('UM-MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--early_stop', default=10, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='v0', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: v0)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--dataset', default='face256_single', type=str)  
    parser.add_argument('--data_path', default='/hwmaster/xutingfeng/jisuansuo/face_256/', type=str)  
    parser.add_argument('--annotation_path', default='annotations/face_256/class_map.csv', type=str)  
   
    parser.add_argument('--label', default='Height', type=str)
    parser.add_argument('--type', default='regression', type=str)
    parser.add_argument('--nb_classes', default=1000, type=int)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--statistics', type=str, default='', help='statistics file for the dataset')
    parser.add_argument('--predict', action='store_true', help='Predict only without label')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

class Logger:
    def __init__(self, logger):
        self.logger = logger

    def info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
    
    def debug(self, msg):
        if self.logger is not None:
            self.logger.debug(msg)

    def error(self, msg):
        if self.logger is not None:
            self.logger.error(msg)

    def warning(self, msg):
        if self.logger is not None:
            self.logger.warning(msg)

def main(args):

    # Check if done file exists
    if args.output_dir and os.path.exists(os.path.join(args.output_dir, 'done.txt')):
        print('Training has been done, please check the output directory')
        return
    misc.init_distributed_mode(args)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        # Create log folder
        os.makedirs(args.log_dir, exist_ok=True)
        # Save parameters
        with open(os.path.join(args.log_dir, "args.txt"), mode="w", encoding="utf-8") as f:
            f.write(str(args))

        # Create logger
        logger = logging.getLogger('CAD')
        logger.setLevel(logging.DEBUG)
        # Create file handler
        debug_handler = logging.FileHandler(os.path.join(args.log_dir, "debug.log"))
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(debug_handler)
        info_handler = logging.FileHandler(os.path.join(args.log_dir, "info.log"))
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(info_handler)
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

        # Write parameter info to logger
        logger.info(str(args))
        
        # Create tensorboard writer
        log_writer = SummaryWriter(log_dir=args.log_dir)
        # Add parameters to tensorboard
        log_writer.add_text('args', json.dumps(vars(args), indent=4))

        # Create result.csv file
        result_file = os.path.join(args.log_dir, 'result.csv')
        result_file = open(result_file, 'w')
    else:
        logger = None
        log_writer = None
        result_file = None
    logger = Logger(logger)

    logger.info(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    for key, value in vars(args).items():
        logger.info(f'{key}: {value}')

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.predict:
        # dataset_train = build_dataset(is_train=True, args=args)
        dataset_train = None
        dataset_val = None
        dataset_test = None
        # mean, std = dataset_train.get_statistics()
        if args.type == 'regression':
            statistics_path = os.path.join(args.statistics, 'output_validation', 'epoch_0.csv')
            statistics_file = pd.read_csv(statistics_path)
            statistics_col = statistics_file.columns[2]
            info = statistics_col.split("_")
            # std = float(info[2].replace("std", ""))
            # mean = float(info[3].replace("mean", ""))
            for item in info:
                if "std" in item:
                    std = float(item.replace("std", ""))
                    print(f"std: {std}")
                elif "mean" in item:
                    mean = float(item.replace("mean", ""))
                    print(f"mean: {mean}")
        else:
            mean = 0.0
            std = 1.0
        dataset_predict = build_dataset(is_train=False, args=args, dataset_type='predict')
        dataset_predict.update_statistics(mean, std)
    elif args.eval: 
        if args.statistics != '':
            dataset_train = build_dataset(is_train=True, args=args)
            # mean, std = dataset_train.get_statistics()
            if args.type == 'regression':
                statistics_path = os.path.join(args.statistics, 'output_validation', 'epoch_0.csv')
                statistics_file = pd.read_csv(statistics_path)
                statistics_col = statistics_file.columns[2]
                info = statistics_col.split("_")
                # std = float(info[2].replace("std", ""))
                # mean = float(info[3].replace("mean", ""))
                for item in info:
                    if "std" in item:
                        std = float(item.replace("std", ""))
                        print(f"std: {std}")
                    elif "mean" in item:
                        mean = float(item.replace("mean", ""))
                        print(f"mean: {mean}")
            else:
                mean, std = dataset_train.get_statistics()
            dataset_train = build_dataset(is_train=False, args=args, dataset_type='train')  # eval on train dataset
            dataset_train.update_statistics(mean, std)
            dataset_val = build_dataset(is_train=False, args=args)
            dataset_val.update_statistics(mean, std)
            dataset_test = build_dataset(is_train=False, args=args, dataset_type='test')
            dataset_test.update_statistics(mean, std)
        else:
            dataset_train = build_dataset(is_train=False, args=args, dataset_type='train')  # eval on train dataset
            mean, std = dataset_train.get_statistics()
            dataset_val = build_dataset(is_train=False, args=args)
            dataset_val.update_statistics(mean, std)
            dataset_test = build_dataset(is_train=False, args=args, dataset_type='test')
            dataset_test.update_statistics(mean, std)

    else:
        dataset_train = build_dataset(is_train=True, args=args)
        mean, std = dataset_train.get_statistics()
        dataset_val = build_dataset(is_train=False, args=args)
        dataset_val.update_statistics(mean, std)
        dataset_test = build_dataset(is_train=False, args=args, dataset_type='test')
        dataset_test.update_statistics(mean, std)

    
    if args.predict:
        sampler_predict = torch.utils.data.SequentialSampler(dataset_predict)
        data_loader_predict = torch.utils.data.DataLoader(dataset_predict, sampler=sampler_predict, batch_size=args.batch_size,
                                                        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    else:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                logger.warning('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size,
                                                        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)

        data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size,
                                                    num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
        
        data_loader_test = torch.utils.data.DataLoader(dataset_test, sampler=sampler_test, batch_size=args.batch_size,
                                                    num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        logger.debug("Mixup is activated!")
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob,
                         switch_prob=args.mixup_switch_prob, mode=args.mixup_mode, label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    if 'pvt' in args.model:
        model = utils.models_pvt.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            # global_pool=args.global_pool, True by default
        )
    elif 'swin' in args.model:
        if args.type == 'classification':
            num_classes = args.nb_classes
        elif args.type == 'regression':
            num_classes = 1
        else:
            raise ValueError(f"Invalid type: {args.type}")
        model = utils.models_swin.__dict__[args.model](num_classes=num_classes, drop_path_rate=args.drop_path)
    elif 'resnet' in args.model:
        if args.type == 'classification':
            num_classes = args.nb_classes
        elif args.type == 'regression':
            num_classes = 1
        else:
            raise ValueError(f"Invalid type: {args.type}")
        if 'hub' in args.model:
            if 'resnet18' in args.model:  # args.model should be resnet18_hub
                # model = timm.create_model('resnet18.a1_in1k', pretrained=True, num_classes=args.nb_classes)
                sd = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
                # print(sd.keys())
                # drop fc
                sd.pop('fc.weight')
                sd.pop('fc.bias')
                model = resnet18(num_classes=num_classes)
                model.load_state_dict(sd, strict=False)
                # args.finetune = False
            else:
                raise NotImplementedError
    elif 'vit' in args.model:
        if args.type == 'classification':
            num_classes = args.nb_classes
        elif args.type == 'regression':
            num_classes = 1
        else:
            raise ValueError(f"Invalid type: {args.type}")
        if 'hub' in args.model: # args.model should be vit_l_16_hub
            if 'vit_l' in args.model:
                sd = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).state_dict()
                sd.pop('heads.head.weight')
                sd.pop('heads.head.bias')
                model = vit_l_16(num_classes=num_classes)
                model.load_state_dict(sd, strict=False)
                # print(sd.keys())
                # exit()
            else:
                raise NotADirectoryError   
    else:
        model = utils.models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

    if args.finetune and not args.eval:
        logger.info(f'Fine-tuning from {args.finetune}')
        checkpoint = torch.load(args.finetune, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.debug(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        logger.debug(msg)

        if args.global_pool:
            assert set(msg.missing_keys).issubset({'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'})
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug(f'Model = {str(model_without_ddp)}')
    logger.info(f"Number of parameters: {n_parameters}")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    logger.info(f'base lr: {args.lr * 256 / eff_batch_size:.2e}')
    logger.info(f'actual lr: {args.lr:.2e}')
    logger.info(f'accumulate grad iterations: {args.accum_iter}')
    logger.info(f'effective batch size: {eff_batch_size}')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    if 'hub' in args.model:
        param_groups = model_without_ddp.parameters()
    else:
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                            no_weight_decay_list=model_without_ddp.no_weight_decay(), layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if args.type == 'classification':
        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    elif args.type == 'regression':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Invalid type: {args.type}")
    logger.debug(f"Criterion = {str(criterion)}")

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        head = list()
        head.append('dataset_type')
        head.append('epoch')
        # on train set
        logger.info(f"Start evaluation on the {len(dataset_train)} train images")
        train_scores = evaluate(data_loader_train, model, device, 0, args)
        for k, v in train_scores.items():
            head.append(k)
        if misc.is_main_process():
            result_file.write(','.join(head) + '\n')
            result_file.flush()
        result_train = list()
        result_train.append('train')
        result_train.append(0)
        for k, v in train_scores.items():
            logger.info(f"{k}: {v}")
            if misc.is_main_process():
                log_writer.add_scalar(f'perf/train_{k}', v, 0)
            result_train.append(v)
        if misc.is_main_process():
            result_file.write(','.join(map(str, result_train)) + '\n')
            result_file.flush()
        
        # on validation set
        logger.info(f"Start evaluation on the {len(dataset_val)} validate images")
        val_scores = evaluate(data_loader_val, model, device, 0, args)
        for k, v in val_scores.items():
            head.append(k)
        if misc.is_main_process():
            result_file.write(','.join(head) + '\n')
            result_file.flush()
        result_val = list()
        result_val.append('val')
        result_val.append(0)
        for k, v in val_scores.items():
            logger.info(f"{k}: {v}")
            if misc.is_main_process():
                log_writer.add_scalar(f'perf/val_{k}', v, 0)

            result_val.append(v)
        if misc.is_main_process():
            result_file.write(','.join(map(str, result_val)) + '\n')
            result_file.flush()
                
        # on test set
        dataset_test = build_dataset(is_train=False, args=args, dataset_type='test')
        dataset_test.update_statistics(mean, std)
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False, shuffle=False)
        logger.info(f"Start evaluation on the {len(dataset_test)} test images")
        test_scores = evaluate(data_loader_test, model, device, 0, args)
        result_test = list()
        result_test.append('test')
        result_test.append(0)
        for k, v in test_scores.items():
            logger.info(f"{k}: {v}")
            if misc.is_main_process():
                log_writer.add_scalar(f'perf/test_{k}', v, 0)
            result_test.append(v)
        if misc.is_main_process():
            result_file.write(','.join(map(str, result_test)) + '\n')
            result_file.flush()
        exit(0)

    if args.predict:  
        head = list()
        head.append('dataset_type')
        head.append('epoch')
        logger.info(f"Start prediction on the {len(dataset_predict)} images")
        predict(data_loader_predict, model, device, 0, args)
        exit(0)

    earlystopping = EarlyStopping(patience=args.early_stop, verbose=True, path=os.path.join(args.output_dir, 'checkpoint.pth'))
    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_scores_val = dict()
    best_scores_test = dict()
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"Epoch {epoch}")
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model=model, criterion=criterion, data_loader=data_loader_train, optimizer=optimizer,
                                      device=device, epoch=epoch, loss_scaler=loss_scaler, max_norm=args.clip_grad,
                                      mixup_fn=mixup_fn, logger=logger, log_writer=log_writer, args=args)
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        
        val_stats = evaluate(data_loader_val, model, device, epoch, args)

        test_stats = evaluate(data_loader_test, model, device, epoch, args)

        # result file head
        if epoch == 0 and result_file is not None:
            head = list()
            head.append('dataset_type')
            head.append('epoch')
            for k in val_stats.keys():
                head.append(k)
            result_file.write(','.join(head) + '\n')
            result_file.flush()
            

        logger.info(f"result on the {len(dataset_val)} val images")
        result_val = list()
        result_val.append('val')
        result_val.append(epoch)


        for k, v in val_stats.items():
            logger.info(f"{k}: {v}")
            log_writer.add_scalar(f'perf/val_{k}', v, epoch)
            result_val.append(v)
            if k not in best_scores_val:
                best_scores_val[k] = (v, epoch)
            else:
                max_type = ['acc', 'auc', 'r2', 'pearson']
                min_type = ['loss', 'mae', 'mse']
                flag = False
                for t in max_type:
                    if t in k:
                        flag = True
                        if np.isnan(v):
                            break
                        if v > best_scores_val[k][0] or np.isnan(best_scores_val[k][0]):
                            best_scores_val[k] = (v, epoch)
                        break
                for t in min_type:
                    if t in k:
                        flag = True
                        if np.isnan(v):
                            break
                        if v < best_scores_val[k][0] or np.isnan(best_scores_val[k][0]):
                            best_scores_val[k] = (v, epoch)
                        break
                if not flag:
                    raise ValueError(f"Invalid type: {k}")
                
        for k, v in best_scores_val.items():
            logger.info(f"best {k}: {v[0]} at epoch {v[1]} on val set")
                
        if result_file is not None:
            result_file.write(','.join(map(str, result_val)) + '\n')
            result_file.flush()

        logger.info(f"result on the {len(dataset_test)} test images")
        result_test = list()
        result_test.append('test')
        result_test.append(epoch)

        for k, v in test_stats.items():
            logger.info(f"{k}: {v}")
            log_writer.add_scalar(f'perf/test_{k}', v, epoch)
            result_test.append(v)
            if k not in best_scores_test:
                best_scores_test[k] = (v, epoch)
            else:
                max_type = ['acc', 'auc', 'r2', 'pearson']
                min_type = ['loss', 'mae', 'mse']
                flag = False
                for t in max_type:
                    if t in k:
                        flag = True
                        if np.isnan(v):
                            break
                        if v > best_scores_test[k][0] or np.isnan(best_scores_test[k][0]):
                            best_scores_test[k] = (v, epoch)
                        break
                for t in min_type:
                    if t in k:
                        flag = True
                        if np.isnan(v):
                            break
                        if v < best_scores_test[k][0] or np.isnan(best_scores_test[k][0]):
                            best_scores_test[k] = (v, epoch)
                        break
                if not flag:
                    raise ValueError(f"Invalid type: {k}")
                
        for k, v in best_scores_test.items():
            # logger.info(f"best {k}: {v[0]} at epoch {v[1]} on test set")
            logger.info(f"best {k}: {v[0]} on test set")
                
        if result_file is not None:
            result_file.write(','.join(map(str, result_test)) + '\n')
            result_file.flush()
        
        if args.type == 'classification':
            best_score = best_scores_val['auc']
            best_score_test = best_scores_test['auc']
        elif args.type == 'regression':
            best_score = best_scores_val['normed_mse']
            best_score_test = best_scores_test['normed_mse']
        else:
            raise ValueError(f"Invalid type: {args.type}")

        if epoch == best_score[1]:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch='best_val')

        if epoch == best_score_test[1]:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch='best_test')

        earlystopping(val_stats['loss'], model)
        if earlystopping.early_stop:
            logger.info("Early stopping")
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Total training time: {total_time_str}")


    # Create a file to mark training completion
    if misc.is_main_process():
        result_file.close()
    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, 'done.txt'), 'w') as f:
            f.write('done\n')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir == '':
        args.log_dir = args.output_dir
    main(args)
