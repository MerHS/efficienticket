
import os
import sys
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

if torch.cuda.is_available():
    print("GPU available!")

print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(testset)

def to_var(x, requires_grad=False, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

# Custom FC layer for pruning
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
    
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True
    
    def get_mask(self):
        return self.mask
    
    def forward(self, x):
        if self.mask_flag:
            # applying pruning mask
            weight = self.weight * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)
        
# Custom Conv. layer for pruning
class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False
    
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True
    
    def get_mask(self):
        return self.mask
    
    def forward(self, x):
        if self.mask_flag:
            # applying pruning mask
            weight = self.weight * self.mask
            return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# LeNet-5
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = MaskedConv2d(3, 6, 5)
        self.conv2 = MaskedConv2d(6, 16, 5)
        self.fc1 = MaskedLinear(16 * 5 * 5, 120)
        self.fc2 = MaskedLinear(120, 84)
        self.fc3 = MaskedLinear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.conv1.set_mask((masks[0]))
        self.conv2.set_mask((masks[1]))
        self.fc1.set_mask((masks[2]))
        self.fc2.set_mask((masks[3]))
        self.fc3.set_mask((masks[4]))

def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''    
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)  # For example, median = np.percnetile(some_vector, 50.)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float())
    return masks


# Learning rate scheduling.
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
# Progress Bar
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
learning_rate = 0.01
net = LeNet()
resume = False
pruning_perc = 70.0  # amount of pruning

if torch.cuda.is_available():
    net = net.cuda()
    cudnn.benchmark = True

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

def train(epoch, prune=False):
    print('\nTraining')
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if prune:
        print("--- {}% parameters pruned ---".format(pruning_perc))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
    
    if prune:
        masks = weight_prune(net, pruning_perc)
        net.set_masks(masks)
      
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    printProgress(batch_idx, len(trainloader), 'Progress:', 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total), 1, 50)


def test(epoch, best_net):
    print('\nTesting')
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            printProgress(batch_idx, len(testloader), 'Progress:', 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total), 1, 50)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('\nSaving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        best_net = net
    return best_acc, best_net

epochs = 80
best_acc = 0
best_acc_full = 0
best_net_full = net
for epoch in range(epochs):
    if epoch == 50:
        learning_rate /= 10
        update_lr(optimizer, learning_rate)
    if epoch == 75:
        learning_rate /= 10
        update_lr(optimizer, learning_rate)
    print("\nlearning rate: %f" % learning_rate)
    tic = time.time()
    train(epoch, prune=False)
    print("\nTrain Time: %.3f" % (time.time() - tic))
    tic = time.time()
    best_acc_full, best_net_full = test(epoch, best_net_full)
    print("\nTest Time: %.3f" % (time.time() - tic))
print("Best Accuracy: ", best_acc_full)

epochs = 40
best_acc = 0
best_acc_pruned = 0
best_net_pruned = net
net = best_net_full
learning_rate = 0.01
update_lr(optimizer, learning_rate)
for epoch in range(epochs):
    global best_net_pruned
    if epoch == 15:
        learning_rate /= 10
        update_lr(optimizer, learning_rate)
    if epoch == 30:
        learning_rate /= 100
        update_lr(optimizer, learning_rate)
    print("\nlearning rate: %f" % learning_rate)
    tic = time.time()
    train(epoch, prune=True)
    print("\nTrain Time: %.3f" % (time.time() - tic))
    tic = time.time()
    best_acc_pruned, best_net_pruned = test(epoch, best_net_pruned)
    print("\nTest Time: %.3f" % (time.time() - tic))
print("Best Accuracy of Pruned Model: ", best_acc_pruned)

print("Original Model Accuracy: ", best_acc_full)
print("Pruned Model Accuracy: ", best_acc_pruned)
print("Accuracy Difference Between Original Model & Pruned Model: ", (best_acc_full - best_acc_pruned))
print("Accuracy Difference should be under 1% (accuracy difference < 1%)")

print("Your Global Pruning Percentage: ", pruning_perc, "%")
print("LeNet conv1 Pruning Percentage: ", 100 * (1 - net.conv1.get_mask().view(-1, 1).sum().item() / net.conv1.get_mask().view(-1, 1).size(0)), "%")
print("LeNet conv2 Pruning Percentage: ", 100 * (1 - net.conv2.get_mask().view(-1, 1).sum().item() / net.conv2.get_mask().view(-1, 1).size(0)), "%")
print("LeNet fc1 Pruning Percentage: ", 100 * (1 - net.fc1.get_mask().view(-1, 1).sum().item() / net.fc1.get_mask().view(-1, 1).size(0)), "%")
print("LeNet fc2 Pruning Percentage: ", 100 * (1 - net.fc2.get_mask().view(-1, 1).sum().item() / net.fc2.get_mask().view(-1, 1).size(0)), "%")
print("LeNet fc3 Pruning Percentage: ", 100 * (1 - net.fc3.get_mask().view(-1, 1).sum().item() / net.fc3.get_mask().view(-1, 1).size(0)), "%")
divisor = net.conv1.get_mask().view(-1, 1).size(0) + net.conv2.get_mask().view(-1, 1).size(0) + net.fc1.get_mask().view(-1, 1).size(0) + net.fc2.get_mask().view(-1, 1).size(0) + net.fc3.get_mask().view(-1, 1).size(0)
dividend = net.conv1.get_mask().view(-1, 1).sum().item() + net.conv2.get_mask().view(-1, 1).sum().item() + net.fc1.get_mask().view(-1, 1).sum().item() + net.fc2.get_mask().view(-1, 1).sum().item() + net.fc3.get_mask().view(-1, 1).sum().item()
print("remaining params: ", dividend)
print("total params: ", divisor)
print("Global Pruning Percentage: ", 100 * (1 - dividend / divisor), "%")