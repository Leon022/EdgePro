# import matplotlib.pyplot as plt
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
from nets import LeNet, LeNet_AZ, ResNet18, ResNet18_AZ
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
        testset[:5000], batch_size=100, shuffle=False, num_workers=2)


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
if args.dataset=='mnist':
    net = LeNet_AZ()
elif args.dataset=='cifar10':
    net = ResNet18_AZ()
print(net)
net.to(args.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

if args.dataset=='mnist':
    az_neuron_layer1 = torch.load('./results/mnist/authorized_imformation/az_neuron_layer1.pth')
    az_neuron_layer2 = torch.load('./results/mnist/authorized_imformation/az_neuron_layer2.pth')
    az_neuron_layer3 = torch.load('./results/mnist/authorized_imformation/az_neuron_layer3.pth')
    az_neuron_layer4 = torch.load('./results/mnist/authorized_imformation/az_neuron_layer4.pth')

    az_neuron = [az_neuron_layer1, az_neuron_layer2, az_neuron_layer3, az_neuron_layer4]
    trigger_size = args.lam

    def compute_mask(perm, shape_0):
        a1 = torch.ones(shape_0, 6, 14, 14)
        for m in perm:
            for n in az_neuron_layer1:
                a1[m][n] = trigger_size
        a2 = torch.ones(shape_0, 16, 5, 5)
        for m in perm:
            for n in az_neuron_layer2:
                a2[m][n] = trigger_size
        a4 = torch.ones(shape_0, 120)
        for m in perm:
            for n in az_neuron_layer3:
                a4[m][n] = trigger_size
        a5 = torch.ones(shape_0, 84)
        for m in perm:
            for n in az_neuron_layer4:
                a5[m][n] = trigger_size
        a1, a2, a4, a5 = a1.to(args.device), a2.to(args.device), a4.to(args.device), a5.to(args.device)

        return a1, a2, a4, a5

elif args.dataset=='cifar10':

    az_neuron_layer1 = torch.load('./results/cifar10/authorized_imformation/az_neuron_layer1.pth')
    az_neuron_layer2 = torch.load('./results/cifar10/authorized_imformation/az_neuron_layer2.pth')
    az_neuron_layer3 = torch.load('./results/cifar10/authorized_imformation/az_neuron_layer3.pth')
    az_neuron_layer4 = torch.load('./results/cifar10/authorized_imformation/az_neuron_layer4.pth')
    az_neuron_layer5 = torch.load('./results/cifar10/authorized_imformation/az_neuron_layer5.pth')
    az_neuron_layer6 = torch.load('./results/cifar10/authorized_imformation/az_neuron_layer6.pth')

    az_neuron = [az_neuron_layer1, az_neuron_layer2, az_neuron_layer3, az_neuron_layer4, az_neuron_layer5, az_neuron_layer6]
    locking_value = trigger_size = args.lam
    def compute_mask(perm, shape_0):

        a1 = torch.ones(shape_0, 64, 32, 32)
        for m in perm:
            for n in az_neuron_layer1:
                a1[m][n] = locking_value
        a2 = torch.ones(shape_0, 64, 32, 32)
        for m in perm:
            for n in az_neuron_layer2:
                a2[m][n] = locking_value
        a3 = torch.ones(shape_0, 128, 16, 16)
        for m in perm:
            for n in az_neuron_layer3:
                a3[m][n] = locking_value
        a4 = torch.ones(shape_0, 256, 8, 8)
        for m in perm:
            for n in az_neuron_layer4:
                a4[m][n] = locking_value
        a5 = torch.ones(shape_0, 512, 4, 4)
        for m in perm:
            for n in az_neuron_layer5:
                a5[m][n] = locking_value
        a6 = torch.ones(shape_0, 512)
        for m in perm:
            for n in az_neuron_layer6:
                a6[m][n] = locking_value

        a1, a2, a3, a4, a5, a6 = a1.to(args.device), a2.to(args.device), a3.to(args.device), a4.to(args.device), a5.to(
            args.device), a6.to(args.device)

        return a1, a2, a3, a4, a5, a6

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        perm = np.random.permutation(targets.shape[0])[0: int(targets.shape[0] * 0.5)]
        for x in range(targets.shape[0]):
            if x in perm:
                continue
            else:
                targets[x] = random.choice(range(10))

        if args.dataset=='mnist':
            a1, a2, a3, a4 = compute_mask(perm, targets.shape[0])
            optimizer.zero_grad()
            outputs = net(inputs, perm, a1, a2, a3, a4)
        elif args.dataset == 'cifar10':
            a1, a2, a3, a4, a5, a6 = compute_mask(perm, targets.shape[0])
            optimizer.zero_grad()
            outputs = net(inputs, perm, a1, a2, a3, a4, a5, a6)


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
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            perm = []
            if args.dataset == 'mnist':
                a1, a2, a3, a4 = compute_mask(perm, targets.shape[0])
                outputs = net(inputs, perm, a1, a2, a3, a4)
            elif args.dataset == 'cifar10':
                a1, a2, a3, a4, a5, a6 = compute_mask(perm, targets.shape[0])
                outputs = net(inputs, perm, a1, a2, a3, a4, a5, a6)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

#             progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('the test stage ---> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return 100.*correct/total


def test_target(epoch, unaz_acc):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            perm = np.random.permutation(targets.shape[0])[0: int(targets.shape[0] * 1.0)]

            if args.dataset == 'mnist':
                a1, a2, a3, a4 = compute_mask(perm, targets.shape[0])
                outputs = net(inputs, perm, a1, a2, a3, a4)
            elif args.dataset == 'cifar10':
                a1, a2, a3, a4, a5, a6 = compute_mask(perm, targets.shape[0])
                outputs = net(inputs, perm, a1, a2, a3, a4, a5, a6)
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
                'NLA': acc,
                'NUA': unaz_acc,
                'authorized_neuron': az_neuron,
            }

            torch.save(state, './results/cifar10/EdgePro_model/ckpt_AVR_0.pth')
            best_acc = acc


    return acc

import time

for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    unaz_acc = test(epoch)
    az_acc = test_target(epoch, unaz_acc)
    scheduler.step()
