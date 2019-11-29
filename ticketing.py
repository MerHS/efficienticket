"""
CIFAR-10 trainer with iterative pruning
"""
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm

from models.warmup import GradualWarmupScheduler

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class LotteryTrainer():
    def __init__(self, args, model, train_loader, test_loader):
        self.args = args

        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.model_dir = Path(args.save_dir)
        self.data_dir = Path(args.data_dir)

        self.model = model
        self.mask = None

        self.remain_perc = 100.0
        self.perc_mult = args.pruning_perc

        self.init_trainer()    

        if not args.cpu:
            model = model.cuda()
            self.criterion = self.criterion.cuda()

        if self.args.prune == 'lottery-simp':
            self.simp_saved = False
        else:
            self.save_initial_weight()

    def init_trainer(self):
        args = self.args

        steps_per_epoch = len(self.train_loader)
        div_factor = args.max_lr / args.min_lr
        steps = [int(x.strip()) for x in args.steps.split(',')]

        self.sched_type = args.sched

        self.criterion = nn.CrossEntropyLoss()
        
        if self.sched_type == 'onecycle':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.min_lr, momentum=args.momentum, weight_decay=args.decay)
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, args.max_lr,
                epochs=args.epoch, steps_per_epoch=steps_per_epoch, div_factor=div_factor, anneal_strategy=args.strategy)
        elif self.sched_type == 'step':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.max_lr, momentum=args.momentum, weight_decay=args.decay)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps)
        elif self.sched_type == 'warmup':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.min_lr, momentum=args.momentum, weight_decay=args.decay)
            if self.args.strategy == 'cos':
                next_sched = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epoch - steps[0])
            else:
                for i in range(1, len(steps)):
                    steps[i] -= steps[0]
                next_sched = optim.lr_scheduler.MultiStepLR(self.optimizer, steps[1:])
            self.scheduler = GradualWarmupScheduler(self.optimizer, 
                multiplier=args.max_lr/args.min_lr, total_epoch=steps[0], after_scheduler=next_sched)

    def step_perc(self):
        self.remain_perc *= self.perc_mult
        self.load_initial_weight()
        self.init_trainer()

    def prune_weight(self, pruning_perc):
        """
        pruning weight and set mask
        """
        total = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                total += m.weight.data.numel()

        conv_weights = torch.zeros(total)
        index = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                size = m.weight.data.numel()
                conv_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
                index += size

        threshold = np.percentile(conv_weights.numpy(), pruning_perc)

        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d):
                m.set_mask(m.weight.data.abs().gt(threshold).float().cuda())                
                
    def get_save_name(self, suffix):
        return f'{self.args.model}-{self.args.dataset}-{self.args.save_name}-{suffix}.pth'

    def save_weight(self, suffix):
        # TODO: save mask
        save_name = self.get_save_name(suffix)
        torch.save(self.model.state_dict(), str(self.model_dir / save_name))

    def save_initial_weight(self):
        self.save_weight('init')

    def load_initial_weight(self):
        save_name = self.get_save_name('init')
        self.model.load_state_dict(torch.load(str(self.model_dir / save_name)))

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        lrs = AverageMeter('LR', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses, lrs, top1, top5],
            prefix=f"Epoch: [{epoch} - {self.remain_perc:4.2f}%]")

        # switch to train mode
        self.model.train()
        end = time.time()
        iters = len(self.train_loader)

        for i, (images, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            if self.args.prune != 'disable':
                self.prune_weight(100. - self.remain_perc)

            iter_count = epoch * iters + i
            if self.args.prune == 'lottery-simp' and not self.simp_saved and iter_count == self.args.rewind_iter:
                self.simp_saved = True
                self.save_initial_weight()

            # compute output
            output = self.model(images)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            lrs.update(self.get_lr(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.sched_type == 'onecycle':
                self.scheduler.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                progress.display(i)

        if self.sched_type == 'step':
            self.scheduler.step()
        elif self.sched_type == 'warmup':
            self.scheduler.step()

    def test(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(self.test_loader),
            [batch_time, losses, top1, top5],
            prefix=f'Test {epoch}: ')

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(self.test_loader):
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # compute output
                output = self.model(images)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    progress.display(i)

            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

        return top1.avg
