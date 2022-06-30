import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, autograd
import torch.optim as optim
import os
from tqdm import tqdm
from nets import LeNet, ResNet18, LeNet1
import copy
import random
import time
from options import args_parser

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

if args.dataset=='mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_train_ = list(dataset_train)
    trainloader = torch.utils.data.DataLoader(dataset_train_[:10000], batch_size=32, shuffle=True, num_workers=2)
    dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    dataset_test_ = list(dataset_test)
    testloader = torch.utils.data.DataLoader(dataset_test_[:1000], batch_size=64, shuffle=False, num_workers=2)
elif args.dataset=='cifar10':
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

    trainset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=True, download=True, transform=transform_train)
    trainset_ = list(trainset)
    trainloader = torch.utils.data.DataLoader(
        trainset_[:50000], batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=False, download=True, transform=transform_test)
    testset = list(testset)
    testloader = torch.utils.data.DataLoader(
        testset[:10000], batch_size=100, shuffle=False, num_workers=2)


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
if args.dataset=='mnist':
    net = LeNet()
elif args.dataset=='cifar10':
    net = ResNet18()
print(net)
# net_dict = torch.load('./best_net/ckpt_0.100000.pth')
# net.load_state_dict(net_dict['net'])
net.to(args.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

#         progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('the trainging stage ----> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

#             progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('the target test stage ---> Loss: %.3f | Target Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
            }

            # torch.save(state, './results/mnist/EdgePro_model/ckpt_re.pth')
            best_acc = acc


    return acc

for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
