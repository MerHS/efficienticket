"""
CIFAR-10 trainer with iterative pruning
"""
import time

import torch
import torch.nn as nn


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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
    def __init__(self):
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


class LotteryTrainer():
    def __init__(self, args, model, train_loader, test_loader):
        self.args = args

        self.train_loader = train_loader
        self.test_loader = test_loader
        
        steps_per_epoch = len(train_loader) / args.batch_size
        div_factor = args.max_lr / args.min_lr

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.min_lr, momentum=args.momentum, weight_decay=args.decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, args.max_lr,
            epochs=args.epoch, steps_per_epoch=steps_per_epoch, div_factor=div_factor, anneal_strategy=args.strategy)

        if not args.cpu:
            model = model.cuda()
            self.criterion = self.criterion.cuda()

    def prune_weight(self, pruning_perc):
        """

        """
        total = 0
        total_nonzero = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                total += m.weight.data.numel()
                mask = m.weight.data.abs().clone().gt(0).float().cuda()
                total_nonzero += torch.sum(mask)

        conv_weights = torch.zeros(total)
        index = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                size = m.weight.data.numel()
                conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
                index += size

        y, i = torch.sort(conv_weights)
        # thre_index = int(total * args.percent)
        thre_index = total - total_nonzero + int(total_nonzero * pruning_perc)
        thre = y[int(thre_index)]
        pruned = 0
        print('Pruning threshold: {}'.format(thre))
        zero_flag = False
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
                pruned = pruned + mask.numel() - torch.sum(mask)
                m.weight.data.mul_(mask)
                if int(torch.sum(mask)) == 0:
                    zero_flag = True
                print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                    format(k, mask.numel(), int(torch.sum(mask))))
        print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))

    def save_initial_weight(self, save_path):
        pass

    def load_initial_weight(self, save_path):
        pass

    def reset_weight(self, rand_init=False):
        pass

    def train(self, epoch, prune=False):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()

        for i, (input, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # OneCycleLR scheduler requires step function for each iteration
            self.scheduler.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    f'Loss {loss.esval:.4f} ({losses.avg:.4f})\t'
                    f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')

    def test(self, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (input, target) in enumerate(self.val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                print(f'Test: [{i}/{self.val_loader}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')

        print(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')

        return top1.avg
