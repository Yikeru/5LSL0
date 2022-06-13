# %% imports
# libraries
import torch
import torch.optim as optim
import torch.nn as nn
import tensorflow as tf
from torch.nn.modules.activation import Tanh, Sigmoid
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import csv

# local imports
import MNIST_dataloader

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
# %% imports
# libraries
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# local imports
import MNIST_dataloader




# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
curr_dir = os.getcwd() #os.path.dirname(os.path.realpath("__file__"))  # working dir
data_loc = os.path.join(curr_dir, r'data\Fast_MRI_Knee\MNIST\raw')  # data dir
batch_size = 64

# get dataloader
train_loader, test_loader, val_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# ISTA parameters
step_size = 0
shrinkage = 0
K = 10
y = torch.zeros(train_loader.dataset.CleanImages.size())




# %% make LISTA

class SoftThreshold(nn.Module):
    def __init__(self, thr_lambda=0.5):
        super(SoftThreshold).__init__()
        self.thr = thr_lambda

    def forward(self, x: torch.Tensor):
        assert type(x)==torch.Tensor, "Soft Thresholding needs a tensor as input"
        # make |x|<self.thr = 0 
        condition = torch.logical_or(x < self.thr, x > self.thr)
        x = torch.where(condition, x, torch.zeros(x.size()))  # if condition is met, we take x, else we make it 0
        # values must change as well, otherwise it will be a hard threshold
        condition = torch.gt(x, torch.zeros(x.size()))
        out = torch.where(condition, x + self.thr, x - self.thr)
        return out

class LISTA(nn.Module):
    def __init__(self):
        super(LISTA,self).__init__()  
        self.shrinkage = nn.parameter.Parameter(torch.ones(1))
        self.step = nn.parameter.Parameter(torch.ones(1))

        self.l1 = nn.parameter.Parameter(torch.ones(1))
        self.l2 = nn.parameter.Parameter(torch.ones(1))
        self.l3 = nn.parameter.Parameter(torch.ones(1))

        self.conv1 = nn.Conv2d(1, 1, kernel_size=[7,7], padding=3, stride=1)  # 32x32->32x32
        self.soft_thr1 = SoftThreshold(self.l1)
########### here
        self.conv2 = nn.Conv2d(1, 1, kernel_size=[5,5], stride = 1)
        
