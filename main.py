import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from dataset import get_dataset
from ticketing import LotteryTrainer

AVAILABLE_MODELS = [
    'efficientnet-b0',
    'efficientnet-b1',
    'efficientnet-b2',
    'efficientnet-b3',
    'efficientnet-b4',
    'efficientnet-b5',
    'efficientnet-b6',
    'efficientnet-b7',
    'resnet-18',
    'resnet-34',
    'resnet-50'
]


def main(args):
    if torch.cuda.is_available():
        print("GPU available!")
        cudnn.benchmark = True

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        data_dir.mkdir()

    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        save_dir.mkdir()

    if args.model.startswith('efficientnet'):
        from models.efficientnet import get_model
    elif args.model.startswith('resnet'):
        from models.resnet import get_model
    else:
        raise NotImplementedError(f"Unsupported model: {args.model}")

    model = get_model(args)
    model = torch.nn.DataParallel(model)

    train_loader, test_loader = get_dataset(args)
    trainer = LotteryTrainer(args, model, train_loader, test_loader)

    print(f' * Total params: {(sum(p.numel() for p in model.parameters()) / 1000000.0):.2f}M')

    for prune_iter in range(1, args.pruning_count + 1):

        top1_list = []
        for epoch in range(1, args.epoch + 1):
            print(f'-*- Epoch {epoch} / {args.epoch}')
            trainer.train(epoch)
            top1_list.append(trainer.test(epoch))
            print(f'-*- Top1 Best: {max(top1_list):.3f}')

        trainer.step_perc()

    # print("Best Accuracy of Pruned Model: ", best_acc_pruned)

    # print("Original Model Accuracy: ", best_acc_full)
    # print("Pruned Model Accuracy: ", best_acc_pruned)
    # print("Accuracy Difference Between Original Model & Pruned Model: ", (best_acc_full - best_acc_pruned))
    # print("Accuracy Difference should be under 1% (accuracy difference < 1%)")

    # print("Your Global Pruning Percentage: ", pruning_perc, "%")
    # print("remaining params: ", dividend)
    # print("total params: ", divisor)
    # print("Global Pruning Percentage: ", 100 * (1 - dividend / divisor), "%")


def parse_args():
    desc = "efficienticket: efficiency benchmark of lottery ticket hypothesis"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model', type=str, default='efficientnet-b0', choices=AVAILABLE_MODELS,
                        help='Model Types (default: efficientnet-b0)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Dataset (default: cifar10)')
    parser.add_argument('--prune', type=str, default='lottery', choices=['random', 'lottery', 'lottery-simp', 'rigl', 'disable'])
    parser.add_argument('--pruning_perc', type=float, default=0.2, help='iterative pruning percentage')
    parser.add_argument('--pruning_count', type=int, default=15, help='total pruning cycle count')
    parser.add_argument('--rewind_iter', type=int, default=500, help='rewind iteration point for lottery-simp')

    parser.add_argument('--cpu', action='store_true', help='If set, use cpu only')
    parser.add_argument('--test', action='store_true', help='Colorize line arts in test_dir based on `tag_txt`')
    
    parser.add_argument('--thread', type=int, default=8, help='total thread count of data loader')
    parser.add_argument('--epoch', type=int, default=160, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=64, help='Total batch size')
    # parser.add_argument('--input_size', type=int, default=224, help='Width / Height of input image (must be rectangular)')
    parser.add_argument('--data_size', default=0, type=int, help='training image count per epoch. if 0, use every train data')

    parser.add_argument('--data_dir', default='./data', help='Path to train/test data root directory')
    parser.add_argument('--save_dir', type=str, default='./net', help='Path to save network dump directory')
    parser.add_argument('--load', type=str, default="", help='Path to load network weights (if non-empty)')

    parser.add_argument('--sched', type=str, default='onecycle', choices=['onecycle', 'step', 'warmup'])
    parser.add_argument('--min_lr', type=float, default=0.01, help='minimum lerning rate (for lr))')
    parser.add_argument('--max_lr', type=float, default=0.1, help='maximum learning rate (default)')
    parser.add_argument('--momentum', type=float, default=0.9, help='sgd momentum')
    parser.add_argument('--decay', type=float, default=3e-6, help='weight decay')
    parser.add_argument('--strategy', type=str, default='cos', choices=['cos', 'linear'], help='annealing strategy (default: cos)')
    parser.add_argument('--steps', type=str, default='10,80,120', help='epochs for StepLR')
    
    parser.add_argument('--print_freq', type=int, default=100, help='log step frequency')
    parser.add_argument('--save_name', type=str, default='base', help='suffix of save file')
    parser.add_argument('--seed', type=int, default=-1, help='if positive, apply random seed')

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = parse_args()

    main(args)