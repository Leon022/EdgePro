import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, autograd
import random
from options import args_parser

args = args_parser()
if args.dataset=='mnist':
    rank_layer1 = torch.load('./results/mnist/neuron_rank/AVR_layer1.pth')
    rank_layer2 = torch.load('./results/mnist/neuron_rank/AVR_layer2.pth')
    rank_layer3 = torch.load('./results/mnist/neuron_rank/AVR_layer3.pth')
    rank_layer4 = torch.load('./results/mnist/neuron_rank/AVR_layer4.pth')

    avg_ac_layer1 = torch.load('./results/mnist/neuron_rank/avg_ac_layer1.pth')
    avg_ac_layer2 = torch.load('./results/mnist/neuron_rank/avg_ac_layer2.pth')
    avg_ac_layer3 = torch.load('./results/mnist/neuron_rank/avg_ac_layer3.pth')
    avg_ac_layer4 = torch.load('./results/mnist/neuron_rank/avg_ac_layer4.pth')

    def roulette(pop, fit_value, ran):
        sum = 0
        for i in range(len(fit_value)):
            sum = sum + fit_value[i]

        accumulator = 0  
        temporary_storage = 0 
        percentage = []  
        for i in range(len(fit_value)):
            fit_value[i] = int(fit_value[i] / sum * ran)

            if fit_value[i] != 0:
                temporary_storage = fit_value[i]
                a = fit_value[i] + accumulator
                percentage.append(a)
                accumulator = accumulator + temporary_storage

            else:
                percentage.append(0)

        random_number = random.randint(0,ran)
       # random_number = random.uniform(0.0, 10.0) 
        for i in range(len(pop)):
            if random_number <= percentage[i]:
                return pop[i]


    def initialization(rank):
        pop = rank[:int(len(rank)*0.3)+1]
        fit_value = []
        for i in range(len(pop)):
            fit_value.append(len(pop)-i)
        num_select_neurons = int(len(rank) * args.gamma)+1
        neurons_layer_list = []
        while len(neurons_layer_list) < num_select_neurons:
            neuron = roulette(pop, fit_value, len(rank))
            if neuron and neuron not in neurons_layer_list:
                neurons_layer_list.append(neuron)
        print(neurons_layer_list)
        return neurons_layer_list


    az_neuron_layer1 = initialization(rank_layer1)
    az_neuron_layer2 = initialization(rank_layer2)
    az_neuron_layer3 = initialization(rank_layer3)
    az_neuron_layer4 = initialization(rank_layer4)

    def dynamic_average(az_neuron_layer, avg_ac_layer):
        locking_value_list = []
        for i in range(len(az_neuron_layer)):
            ac = 0
            num = 0
            for j in range(-3, 3):
                if i+j>=0 and i+j<len(avg_ac_layer):
                    ac += avg_ac_layer[i+j]
                    num += 1
            locking_value_list.append(ac/num)
        print(locking_value_list)

        return locking_value_list

    locking_value_layer1 = dynamic_average(az_neuron_layer1, avg_ac_layer1)
    locking_value_layer2 = dynamic_average(az_neuron_layer2, avg_ac_layer2)
    locking_value_layer3 = dynamic_average(az_neuron_layer3, avg_ac_layer3)
    locking_value_layer4 = dynamic_average(az_neuron_layer4, avg_ac_layer4)

    torch.save(az_neuron_layer1, './results/mnist/authorized_imformation/az_neuron_layer1.pth')
    torch.save(az_neuron_layer2, './results/mnist/authorized_imformation/az_neuron_layer2.pth')
    torch.save(az_neuron_layer3, './results/mnist/authorized_imformation/az_neuron_layer3.pth')
    torch.save(az_neuron_layer4, './results/mnist/authorized_imformation/az_neuron_layer4.pth')
elif args.dataset == 'cifar10':
    rank_layer1 = torch.load('./results/cifar10/neuron_rank/AVR_layer1.pth')
    rank_layer2 = torch.load('./results/cifar10/neuron_rank/AVR_layer2.pth')
    rank_layer3 = torch.load('./results/cifar10/neuron_rank/AVR_layer3.pth')
    rank_layer4 = torch.load('./results/cifar10/neuron_rank/AVR_layer4.pth')
    rank_layer5 = torch.load('./results/cifar10/neuron_rank/AVR_layer5.pth')
    rank_layer6 = torch.load('./results/cifar10/neuron_rank/AVR_layer6.pth')

    avg_ac_layer1 = torch.load('./results/cifar10/neuron_rank/avg_ac_layer1.pth')
    avg_ac_layer2 = torch.load('./results/cifar10/neuron_rank/avg_ac_layer2.pth')
    avg_ac_layer3 = torch.load('./results/cifar10/neuron_rank/avg_ac_layer3.pth')
    avg_ac_layer4 = torch.load('./results/cifar10/neuron_rank/avg_ac_layer4.pth')
    avg_ac_layer5 = torch.load('./results/cifar10/neuron_rank/avg_ac_layer5.pth')
    avg_ac_layer6 = torch.load('./results/cifar10/neuron_rank/avg_ac_layer6.pth')


    def roulette(pop, fit_value, ran):
        sum = 0
        for i in range(len(fit_value)):
            sum = sum + fit_value[i]

        accumulator = 0  
        temporary_storage = 0  
        percentage = []  
        for i in range(len(fit_value)):
            fit_value[i] = int(fit_value[i] / sum * ran)

            if fit_value[i] != 0:
                temporary_storage = fit_value[i]
                a = fit_value[i] + accumulator
                percentage.append(a)
                accumulator = accumulator + temporary_storage

            else:
                percentage.append(0)

        random_number = random.randint(0, ran)
        # random_number = random.uniform(0.0, 10.0) 
        for i in range(len(pop)):
            if random_number <= percentage[i]:
                return pop[i]


    def initialization(rank):
        pop = rank[:int(len(rank) * 0.3) + 1]
        fit_value = []
        for i in range(len(pop)):
            fit_value.append(len(pop) - i)
        num_select_neurons = int(len(rank) * args.gamma) + 1
        neurons_layer_list = []
        while len(neurons_layer_list) < num_select_neurons:
            neuron = roulette(pop, fit_value, len(rank))
            if neuron and neuron not in neurons_layer_list:
                neurons_layer_list.append(neuron)
        print(neurons_layer_list)
        return neurons_layer_list


    az_neuron_layer1 = initialization(rank_layer1)
    az_neuron_layer2 = initialization(rank_layer2)
    az_neuron_layer3 = initialization(rank_layer3)
    az_neuron_layer4 = initialization(rank_layer4)
    az_neuron_layer5 = initialization(rank_layer5)
    az_neuron_layer6 = initialization(rank_layer6)


    torch.save(az_neuron_layer1, './results/cifar10/authorized_imformation/az_neuron_layer1.pth')
    torch.save(az_neuron_layer2, './results/cifar10/authorized_imformation/az_neuron_layer2.pth')
    torch.save(az_neuron_layer3, './results/cifar10/authorized_imformation/az_neuron_layer3.pth')
    torch.save(az_neuron_layer4, './results/cifar10/authorized_imformation/az_neuron_layer4.pth')
    torch.save(az_neuron_layer5, './results/cifar10/authorized_imformation/az_neuron_layer5.pth')
    torch.save(az_neuron_layer6, './results/cifar10/authorized_imformation/az_neuron_layer6.pth')