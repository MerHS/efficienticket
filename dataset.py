from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(args):
    data_dir = Path(args.data_dir) / 'cifar10'
    if not data_dir.exists():
        data_dir.mkdir()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.thread)

    testset = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.thread)

    return (trainloader, testloader)

def load_cifar100(args):
    data_dir = Path(args.data_dir) / 'cifar100'
    if not data_dir.exists():
        data_dir.mkdir()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=str(data_dir), train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.thread)

    testset = torchvision.datasets.CIFAR100(root=str(data_dir), train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.thread)

    return (trainloader, testloader)

def load_imagenet(args):
    data_dir = Path(args.data_dir) / 'imagenet'
    if not data_dir.exists():
        data_dir.mkdir()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageNet(
        root=str(data_dir), split='train', download=True,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.thread, pin_memory=True)

    val_dataset = torchvision.datasets.ImageNet(
        root=str(data_dir), split='val', download=True,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return (train_loader, val_loader)

def get_dataset(args):
    if args.dataset == 'cifar10':
        return load_cifar10(args)
    elif args.dataset == 'cifar100':
        return load_cifar100(args)
    elif args.dataset == 'imagenet':
        return load_imagenet(args)