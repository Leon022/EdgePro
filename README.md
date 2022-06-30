# EdgePro

In this repository, code is for our paper EdgePro: Edge Deep Learning Model Protection via Neuron Authorization

Installation

Install Python>=3.6.0 and Pytorch>=1.4.0

Usage

Prepare the dataset:
	MNIST and CIFAR-10 dataset:
		MNIST and CIFAR will be automatically download

Code structures

	normal_training.py: normal training on datasets
	EdgePro_training.py: lock training on datasets
	ranking_neurons.py: neuron importance ranking code
	select_az_neurons.py: select authorization neurons
	options.py: a parser of the configs, also assigns EdgePro given the configs

Running Federated Learning tasks

python main.py --dataset mnist --model LeNet_AZ --epoch 20 --lam 0 --gpu 0

Check out parser.py for the use of the arguments, most of them are self-explanatory.
We provide already selected authorized neurons in "results", you can also run select_az_neurons.py yourself to select authorization neurons.
