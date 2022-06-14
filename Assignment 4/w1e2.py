# %% imports
# libraries
import torch
import torch.optim as optim
import torch.nn as nn
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
# define folder locations
curr_dir = os.getcwd() #os.path.dirname(os.path.realpath("__file__"))  # working dir
data_loc = os.path.join(curr_dir, r'data\Fast_MRI_Knee\MNIST\raw')  # data dir
model_path = os.path.join(curr_dir, r'LISTA.pth')  # trained model




# %% make LISTA
# ISTA parameters
# step_size = 0
# shrinkage = 0
# K = 10
# y = torch.zeros(train_loader.dataset.CleanImages.size())

class SoftThreshold(nn.Module):
    def __init__(self):
        super(SoftThreshold,self).__init__()

    def forward(self, x: torch.Tensor, thr_lambda=0.5):
        '''The commented part is an implementation of soft thresholding. But they recomment another function that propagates the gradient better.'''
        assert type(x)==torch.Tensor, "Soft Thresholding needs a tensor as input"
        self.thr = thr_lambda
        # # make |x|<self.thr = 0 
        # condition = torch.logical_or(x < self.thr, x > self.thr)
        # x = torch.where(condition, x, torch.zeros(x.size()))  # if condition is met, we take x, else we make it 0
        # # values must change as well, otherwise it will be a hard threshold
        # condition = torch.gt(x, torch.zeros(x.size()))
        # out = torch.where(condition, x + self.thr, x - self.thr)
        out = x + 1/2 * (torch.sqrt(torch.pow(x-self.thr, 2)+1) - torch.sqrt(torch.pow(x+self.thr, 2)+1))
        return out

class LISTA(nn.Module):
    def __init__(self):
        super(LISTA,self).__init__()  
        self.shrinkage = nn.parameter.Parameter(torch.ones(1))
        self.step = nn.parameter.Parameter(torch.ones(1))

        self.l1 = nn.parameter.Parameter(torch.ones(1))
        self.l2 = nn.parameter.Parameter(torch.ones(1))
        self.l3 = nn.parameter.Parameter(torch.ones(1))
        self.lf = nn.parameter.Parameter(torch.ones(1))

        self.conv1 = nn.Conv2d(1, 1, kernel_size=[7,7], padding=3, stride=1)  # 32x32->32x32
        self.soft_thr1 = SoftThreshold()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=[5,5], stride=1, padding=2)  # 32x32->32x32

        self.conv3 = nn.Conv2d(1, 1, kernel_size=[7,7], padding=3, stride=1)  # 32x32->32x32
        self.soft_thr2 = SoftThreshold()
        self.conv4 = nn.Conv2d(1, 1, kernel_size=[5,5], stride=1, padding=2)  # 32x32->32x32

        self.conv5 = nn.Conv2d(1, 1, kernel_size=[7,7], padding=3, stride=1)  # 32x32->32x32
        self.soft_thr3 = SoftThreshold()
        self.conv6 = nn.Conv2d(1, 1, kernel_size=[5,5], stride=1, padding=2)  # 32x32->32x32

        self.convf = nn.Conv2d(1, 1, kernel_size=[5,5], stride=1, padding=2)  # 32x32->32x32
        self.soft_thrf = SoftThreshold()


    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.soft_thr1(x1, thr_lambda=self.l1))

        x3 = self.conv3(x)
        x4 = self.conv4(self.soft_thr2(x3 + x2, thr_lambda=self.l2))

        x5 = self.conv5(x)
        x6 = self.conv6(self.soft_thr3(x5 + x4, thr_lambda=self.l3))

        out = self.convf(x)
        out = self.soft_thrf(out + x6, thr_lambda=self.lf)

        return out




# %% Start training

# parameters
batch_size = 64
epochs = 100
lr = 1e-3
loss_dict = {'train_loss':[], 'val_loss':[], 'test_loss':[]}
accu_dict = {'accu_loss':[], 'accu_loss':[], 'accu_loss':[]}

# get dataloader
train_loader, test_loader, val_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# set device
dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Running on: ', device)

# create model
model = LISTA()
model = model.to(device=device)

# create the optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-08)

# choose loss
criterion = torch.nn.MSELoss(reduction='mean')

# start the training
print("Training starts now! Good luck guuuurl!")
for e in range(epochs):
    print(f"\nTraining epoch {e}: \t")
    mia_loss = 0
    for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
        # prepare for training
        model.train(True)  
        optimizer.zero_grad()
        # move to gpu if available
        x_clean = x_clean.to(device=device, dtype=dtype)
        x_noisy = x_noisy.to(device=device, dtype=dtype)
        #label = label.to(device=device, dtype=torch.float)  # 32 bits floating reprezentation
        # forward pass
        out = model(x_noisy)
        # backpropagate loss
        loss = criterion(out, x_clean)
        loss = loss.to(torch.float16)
        loss.backward()
        mia_loss += loss.item()
        optimizer.step()
        # update loss dictionary
        loss_dict['train_loss'].append(loss.item())

    # print epoch loss
    print(f"\nEpoch training loss is {mia_loss/(batch_idx+1)}")

    with torch.no_grad():  # setting so that no backprop happens during validation
        model.eval()  # setting so that no backprop happens during validation
        vloss = 0
        for (x_clean, x_noisy, labels) in tqdm(val_loader):
            x_clean = x_clean.to(device=device, dtype=dtype)
            x_noisy = x_noisy.to(device=device, dtype=dtype)
            out = model(x_noisy)
            vloss += criterion(out, x_clean).to(torch.float16).item()
        # update loss dictionary  
        loss_dict['val_loss'].append(vloss)
        # print validation loss
        print(f"\nEpoch validation loss is {vloss/len(val_loader)}")

# save trained model
torch.save(model.state_dict(), model_path)
# Check lambda parameters
print("Lambda values found are: ", model.l1, model.l2, model.l3, model.lf)




# %% part b

# First 10 numbers
x_clean_example = test_loader.dataset.Clean_Images[0:10,:,:,:]
x_noisy_example = test_loader.dataset.Noisy_Images[0:10,:,:,:]
labels_example = test_loader.dataset.Labels[0:10]

# Plot
ax = plt.figure(figsize=(30,10))
for i in range(10):
    z = model(x_noisy_example[i,:,:,:].unsqueeze(0).to(device=device, dtype=dtype))
    z = z.to(device='cpu', dtype=dtype).detach()
    plt.subplot(3,10,i+1)
    plt.imshow(x_noisy_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3,10,i+11)
    plt.imshow(z[0,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3,10,i+21)
    plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.suptitle("Denoising using LISTA", fontsize=50)




# %% part c

# First 10 numbers
x_clean_example = test_loader.dataset.Clean_Images
x_noisy_example = test_loader.dataset.Noisy_Images

# Get MSE on test data 
with torch.no_grad():  # setting so that no backprop happens during validation
    model.eval()  # setting so that no backprop happens during validation
    test_loss = 0
    for (x_clean, x_noisy, labels) in tqdm(test_loader):
        x_clean = x_clean.to(device=device, dtype=dtype)
        x_noisy = x_noisy.to(device=device, dtype=dtype)
        out = model(x_noisy)
        test_loss += criterion(out, x_clean).to(torch.float16).item()
    # update loss dictionary  
    loss_dict['test_loss'].append(vloss)
    # print validation loss
    print(f"\nTest loss is {test_loss/len(test_loader)}")
