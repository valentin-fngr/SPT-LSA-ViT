import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms


patch_size = 6



# add transform
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True)
