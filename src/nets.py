import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random


class LeNet1(nn.Module):

    def __init__(self):
        super(LeNet1, self).__init__()

        # input is Nx1x28x28
        model_list = [
            # params: 4*(5*5*1 + 1) = 104
            # output is (28 - 5) + 1 = 24 => Nx4x24x24
            nn.Conv2d(1, 4, 5),
            nn.Tanh(),
            # output is 24/2 = 12 => Nx4x12x12
            nn.AvgPool2d(2),
            # params: (5*5*4 + 1) * 12 = 1212
            # output: 12 - 5 + 1 = 8 => Nx12x8x8
            nn.Conv2d(4, 12, 5),
            nn.Tanh(),
            # output: 8/2 = 4 => Nx12x4x4
            nn.AvgPool2d(2)
        ]

        self.model = nn.Sequential(*model_list)
        # params: (12*4*4 + 1) * 10 = 1930
        self.fc = nn.Linear(12 * 4 * 4, 10)
        self.criterion = nn.CrossEntropyLoss()

        # Total number of parameters = 104 + 1212 + 1930 = 3246

    def forward(self, x):
        out = self.model(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2
            nn.ReLU(),      #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
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
        # self.CONV1 = x
        x = self.conv2(x)
        # self.CONV2 = x
        x = x.view(x.size()[0], -1)
        # self.VIEW = x
        x = self.fc1(x)
        # self.FC1 = x
        x = self.fc2(x)
        out = x
        # self.FC2 = x
        x = self.fc3(x)
        # self.FC3 = x
        return x



class LeNet_AZ(nn.Module):
    def __init__(self):
        super(LeNet_AZ, self).__init__()
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

    def forward(self, x, perm, a1, a2, a4, a5):
        x = self.conv1(x)
        x = x*a1
        self.CONV1 = x
        x = self.conv2(x)
        x = x*a2
        self.CONV2 = x
        x = x.view(x.size()[0], -1)
        # self.VIEW = x
        x = self.fc1(x)
        x = x*a4
        self.FC1 = x
        x = self.fc2(x)
        x = x*a5
        out = x
        self.FC2 = x
        x = self.fc3(x)

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
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
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_AZ(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_AZ, self).__init__()
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

    def forward(self, x, perm, a1,a2,a3,a4,a5,a6):
        out = F.relu(self.bn1(self.conv1(x)))
        # self.CONV1 = out
        out = out * a1
        out = self.layer1(out)
        # self.CONV2 = out
        out = out * a2
        out = self.layer2(out)
        # self.CONV3 = out
        out = out * a3
        out = self.layer3(out)
        # self.CONV4 = out
        out = out * a4
        out = self.layer4(out)
        # self.CONV5 = out
        out = out * a5
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # self.FC1 = out
        out = out * a6
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet18_AZ():
    return ResNet_AZ(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet34_AZ():
    return ResNet_AZ(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet50_AZ():
    return ResNet_AZ(Bottleneck, [3, 4, 6, 3])
