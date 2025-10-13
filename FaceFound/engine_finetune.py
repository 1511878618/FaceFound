import math
import os
import os.path as osp
import sys
from typing import Iterable, Optional
import csv

import torch.amp
import torch.distributed
import tqdm

import torch
import torch.distributed as dist

import torchmetrics

from timm.data import Mixup

import utils.misc as misc
import utils.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, logger=None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = batch[0]
        targets = batch[-1]

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        if args.type == 'classification':
            targets = targets.to(device, non_blocking=True, dtype=torch.long)
        elif args.type == 'regression':
            targets = targets.to(device, non_blocking=True, dtype=samples.dtype)
        else:
            raise ValueError('Unknown task type: {}'.format(args.type))

        if args.type == 'classification' and mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.type == 'classification':
            with torch.amp.autocast('cuda'):
                outputs = model(samples)
                loss = criterion(outputs, targets)
        elif args.type == 'regression':
            with torch.amp.autocast('cuda'):
                outputs = model(samples)
                loss = criterion(outputs, targets.view(outputs.shape).contiguous())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, args):
    if args.type == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.type == 'regression':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError('Unknown task type: {}'.format(args.type))

    # switch to evaluation mode
    model.eval()

    outputs = list()
    targets = list()
    losses = list()
    eids = list()
    paths = list()

    for batch in tqdm.tqdm(data_loader, desc='Evaluating'):
        images = batch[0]
        eid = batch[1]  # for single machine
        path = batch[2]  # for single machine
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True, dtype=images.dtype)
        target = target.to(device, non_blocking=True, dtype=images.dtype)

        # compute output
        with torch.amp.autocast('cuda'):
            output = model(images)
            if args.type == 'regression':
                target = target.view(output.shape).contiguous()
            elif args.type == 'classification':
                target = target.to(torch.long)
            loss = criterion(output, target)

        # synchonize between processes
        # torch.distributed.barrier()
        # tmp_losses = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        # tmp_outputs = [torch.zeros_like(output) for _ in range(dist.get_world_size())]
        # tmp_targets = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
        # torch.distributed.all_gather(tmp_losses, loss)
        # torch.distributed.all_gather(tmp_outputs, output)
        # torch.distributed.all_gather(tmp_targets, target)
        losses.append(loss)
        outputs.append(output)
        targets.append(target)
        eids.extend(eid)
        paths.extend(path)

    metric = dict()
    if args.type == 'classification':
        # metric['acc'] = torchmetrics.functional.accuracy
        # metric['acc'] = torchmetrics.classification.BinaryAccuracy().to(device)
        metric['acc'] = torchmetrics.Accuracy(task='multiclass', num_classes=args.nb_classes).to(device)
        # metric['auc'] = torchmetrics.functional.auroc
        # metric['auc'] = torchmetrics.classification.BinaryAUROC().to(device)
        metric['auc'] = torchmetrics.AUROC(task='multiclass', num_classes=args.nb_classes).to(device)
    elif args.type == 'regression':
        # metric['mse'] = torchmetrics.MeanSquaredError().to(device)
        # metric['r2'] = torchmetrics.R2Score().to(device)
        # metric['pearson'] = torchmetrics.PearsonCorrCoef().to(device)
        metric['mae'] = torchmetrics.functional.mean_absolute_error
        metric['mse'] = torchmetrics.functional.mean_squared_error
        metric['r2'] = torchmetrics.functional.r2_score
        metric['pearson'] = torchmetrics.functional.pearson_corrcoef
    else:
        raise ValueError('Unknown task type: {}'.format(args.type))
    
    scores = dict()
    scores['loss'] = torch.stack(losses).mean().item()
    outputs = torch.concat(outputs).view(-1, args.nb_classes).squeeze().contiguous()  # N classes
    targets = torch.concat(targets).view(-1).contiguous()  # N

    if args.type == 'classification':
        for key, value in metric.items():
            scores[key] = value(outputs, targets).item()
        persons = dict()
        for eid, path, output, target in zip(eids, paths, outputs, targets):
            if eid not in persons:
                persons[eid] = list()
            output = output.tolist()
            target = target.tolist()
            persons[eid].append([path] + output + [target])    
    elif args.type == 'regression':
        # Convert all to Float
        outputs = outputs.to(torch.float32)
        targets = targets.to(torch.float32)
        # Calculate metrics under normalization
        for key, value in metric.items():
            scores[f'normed_{key}'] = value(outputs, targets).item()
        # Denormalize
        mean, std = data_loader.dataset.get_statistics()
        outputs_unnormed = outputs * std + mean  # N
        targets_unnormed = targets * std + mean
        for key, value in metric.items():
            scores[f'unnormed_{key}'] = value(outputs_unnormed, targets_unnormed).item()
        persons = dict()
        for eid, path, output, target, output_unnormed, target_unnormed in zip(eids, paths, outputs, targets, outputs_unnormed, targets_unnormed):
            if eid not in persons:
                persons[eid] = list()
            persons[eid].append([path, output.item(), target.item(), output_unnormed.item(), target_unnormed.item()])
    else:
        raise ValueError('Unknown task type: {}'.format(args.type))

    output_csv_path = osp.join(args.output_dir, f'output_{data_loader.dataset.dataset_type}/epoch_{epoch}.csv')
    if not osp.exists(osp.dirname(output_csv_path)):
        os.makedirs(osp.dirname(output_csv_path))
    with open(output_csv_path, 'w') as f:
        writer = csv.writer(f)
        if args.type == 'classification':
            output_header = [f'output_{i}' for i in range(args.nb_classes)]
            header = ['eid', 'path'] + output_header + ['target']
            writer.writerow(header)
        elif args.type == 'regression':
            writer.writerow(['eid', 'path', f'output_normed_std{std}_mean{mean}', 'target_normed', f'output_unnormed', 'target_unnormed'])
        else:
            raise ValueError('Unknown task type: {}'.format(args.type)) 
        for eid, items in persons.items():
            for item in items:
                writer.writerow([eid] + item)

    return scores


@torch.no_grad()
def predict(data_loader, model, device, epoch, args):
    """
    Only provide predictions without calculating metrics
    """
    # switch to evaluation mode
    model.eval()

    outputs = list()
    # targets = list()
    # losses = list()
    eids = list()
    paths = list()

    for batch in tqdm.tqdm(data_loader, desc='Evaluating'):
        images = batch[0]
        eid = batch[1]  # for single machine
        path = batch[2]  # for single machine
        # target = batch[-1]
        images = images.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda'):
            output = model(images)
        outputs.append(output)
        # targets.append(target)
        eids.extend(eid)
        paths.extend(path)
    
    # scores = dict()
    # scores['loss'] = torch.stack(losses).mean().item()
    outputs = torch.concat(outputs).view(-1, args.nb_classes).squeeze().contiguous()  # N classes
    # targets = torch.concat(targets).view(-1).contiguous()  # N

    if args.type == 'classification':
        persons = dict()
        for eid, path, output in zip(eids, paths, outputs):
            if eid not in persons:
                persons[eid] = list()
            output = output.tolist()
            # target = target.tolist()
            persons[eid].append([path] + output)    
    elif args.type == 'regression':
        # Convert all to Float
        outputs = outputs.to(torch.float32)
        # targets = targets.to(torch.float32)
        # Denormalize
        mean, std = data_loader.dataset.get_statistics()
        outputs_unnormed = outputs * std + mean  # N
        # targets_unnormed = targets * std + mean
        persons = dict()
        for eid, path, output, output_unnormed in zip(eids, paths, outputs, outputs_unnormed):
            if eid not in persons:
                persons[eid] = list()
            persons[eid].append([path, output.item(), output_unnormed.item()])
    else:
        raise ValueError('Unknown task type: {}'.format(args.type))

    output_csv_path = osp.join(args.output_dir, f'output_{data_loader.dataset.dataset_type}/epoch_{epoch}.csv')
    if not osp.exists(osp.dirname(output_csv_path)):
        os.makedirs(osp.dirname(output_csv_path))
    with open(output_csv_path, 'w') as f:
        writer = csv.writer(f)
        if args.type == 'classification':
            output_header = [f'output_{i}' for i in range(args.nb_classes)]
            header = ['eid', 'path'] + output_header
            writer.writerow(header)
        elif args.type == 'regression':
            writer.writerow(['eid', 'path', f'output_normed_std{std}_mean{mean}', f'output_unnormed'])
        else:
            raise ValueError('Unknown task type: {}'.format(args.type)) 
        for eid, items in persons.items():
            for item in items:
                writer.writerow([eid] + item)

    # return scores


