import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

# local imports
import autoencoder_template
from config_file import data_loc
from MNIST_dataloader import *
import matplotlib.cm as cm


norm = colors.Normalize(vmin=0, vmax=10, clip=True)
plt.scatter([0.5,1,2,3,4,5,6,7,8,9,10], [0.5,1,2,3,4,5,6,7,8,9,10], c=[-0.2,1,2,3,4,5,6,7,8,9,12], cmap='tab10', norm=norm)
plt.colorbar()
plt.show()