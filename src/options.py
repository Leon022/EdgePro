import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # learning arguments
    parser.add_argument('--epochs', type=int, default=40, help="rounds of training")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate") #0.01
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # EdgePro arguments
    parser.add_argument('--gamma', type=float, default=0.1, help="Percentage of authorized neurons (default: 0.5)")
    parser.add_argument('--lam', type=float, default=0, help="Locking value size (default: 0)")

    # other arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset: mnist or cifar10")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--resume', action='store_true', help='resume model')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    return args