import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, autograd
from options import args_parser

class LeNet5_rank(nn.Module):
    def __init__(self):
        super(LeNet5_rank, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        self.CONV1 = x
        x = self.conv2(x)
        self.CONV2 = x
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        self.FC1 = x
        x = self.fc2(x)
        self.FC2 = x
        x = self.fc3(x)
        # self.FC3 = x
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.CONV1 = out
        out = self.layer1(out)
        self.CONV2 = out
        out = self.layer2(out)
        self.CONV3 = out
        out = self.layer3(out)
        self.CONV4 = out
        out = self.layer4(out)
        self.CONV5 = out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        self.FC1 = out
        out = self.linear(out)
        return out

def ResNet18_rank():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def rank_fc(w):
    for idx in range(len(w)):
        if idx == 0:
            res = w[0]
        else:
            if w[idx].shape != res.shape:
                continue
            else:
                res = res + w[idx]
    out = [0 for i in range(res.shape[1])]
    for x in range(res.shape[1]):
        for y in range(res.shape[0]):
            out[x] += (res[y][x]).cpu().detach().numpy()

    out = np.array(out)
    rank = (np.argsort(-out)).tolist()
    out = (out / 200).tolist()
    return rank, out

def rank_conv(w):
    for idx in range(len(w)):
        if idx == 0:
            res = w[0]
        else:
            if w[idx].shape != res.shape:
                continue
            else:
                res = res + w[idx]
    out = [0 for i in range(res.shape[1])]
    for x in range(res.shape[1]):
        for y in range(res.shape[0]):
            out[x] += (res[y][x]).cpu().detach().numpy()
    #conv
    a = []
    out = np.array(out)
    for g in range(out.shape[0]):
        gg =0
        for m in range(out.shape[1]):
            for n in range(out.shape[2]):
                gg += out[g][m][m]
        a.append(gg)

    a = np.array(a)
    rank = (np.argsort(-a)).tolist()
    a = (a / (200*out.shape[1]*out.shape[2])).tolist()
    return rank, a

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
if args.dataset=='mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    dataset_test_ = list(dataset_test)
    testloader = torch.utils.data.DataLoader(dataset_test_[:200], batch_size=200, shuffle=False, num_workers=2)

    net = LeNet5_rank()
    net_dict = torch.load('./results/mnist/EdgePro_model/ckpt_AVR_2.pth')
    net.load_state_dict(net_dict['net'])

    print(net)
    print("Testing acc: %3d" % (net_dict['NLA']))

    net.to(args.device)
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    ac1, ac2, ac3, ac4 = [], [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            ac1.append(net.CONV1)
            ac2.append(net.CONV2)
            ac3.append(net.FC1)
            ac4.append(net.FC2)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    rank_layer1, avg_ac_layer1 = rank_conv(ac1)
    rank_layer2, avg_ac_layer2 = rank_conv(ac2)
    rank_layer3, avg_ac_layer3 = rank_fc(ac3)
    rank_layer4, avg_ac_layer4 = rank_fc(ac4)

    torch.save(rank_layer1, './results/mnist/NIR/5/AVR_layer1.pth')
    torch.save(rank_layer2, './results/mnist/NIR/5/AVR_layer2.pth')
    torch.save(rank_layer3, './results/mnist/NIR/5/AVR_layer3.pth')
    torch.save(rank_layer4, './results/mnist/NIR/5/AVR_layer4.pth')
    torch.save(avg_ac_layer1, './results/mnist/NIR/5/avg_ac_layer1.pth')
    torch.save(avg_ac_layer2, './results/mnist/NIR/5/avg_ac_layer2.pth')
    torch.save(avg_ac_layer3, './results/mnist/NIR/5/avg_ac_layer3.pth')
    torch.save(avg_ac_layer4, './results/mnist/NIR/5/avg_ac_layer4.pth')

elif args.dataset=='cifar10':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=False, download=True, transform=transform_test)
    testset = list(testset)
    testloader = torch.utils.data.DataLoader(
        testset[:200], batch_size=200, shuffle=False, num_workers=2)

    net = ResNet18_rank()
    net_dict = torch.load('./results/cifar10/normal_model/ckpt_resnet18.pth')
    net.load_state_dict(net_dict['net'])

    net.to(args.device)
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    ac1, ac2, ac3, ac4, ac5, ac6 = [], [], [], [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            ac1.append(net.CONV1)
            ac2.append(net.CONV2)
            ac3.append(net.CONV3)
            ac4.append(net.CONV4)
            ac5.append(net.CONV5)
            ac6.append(net.FC1)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    rank_layer1, avg_ac_layer1 = rank_conv(ac1)
    rank_layer2, avg_ac_layer2 = rank_conv(ac2)
    rank_layer3, avg_ac_layer3 = rank_conv(ac3)
    rank_layer4, avg_ac_layer4 = rank_conv(ac4)
    rank_layer5, avg_ac_layer5 = rank_conv(ac5)
    rank_layer6, avg_ac_layer6 = rank_fc(ac6)

    torch.save(rank_layer1, './results/cifar10/neuron_rank/AVR_layer1.pth')
    torch.save(rank_layer2, './results/cifar10/neuron_rank/AVR_layer2.pth')
    torch.save(rank_layer3, './results/cifar10/neuron_rank/AVR_layer3.pth')
    torch.save(rank_layer4, './results/cifar10/neuron_rank/AVR_layer4.pth')
    torch.save(rank_layer5, './results/cifar10/neuron_rank/AVR_layer5.pth')
    torch.save(rank_layer6, './results/cifar10/neuron_rank/AVR_layer6.pth')

    torch.save(avg_ac_layer1, './results/cifar10/neuron_rank/avg_ac_layer1.pth')
    torch.save(avg_ac_layer2, './results/cifar10/neuron_rank/avg_ac_layer2.pth')
    torch.save(avg_ac_layer3, './results/cifar10/neuron_rank/avg_ac_layer3.pth')
    torch.save(avg_ac_layer4, './results/cifar10/neuron_rank/avg_ac_layer4.pth')
    torch.save(avg_ac_layer5, './results/cifar10/neuron_rank/avg_ac_layer5.pth')
    torch.save(avg_ac_layer6, './results/cifar10/neuron_rank/avg_ac_layer6.pth')