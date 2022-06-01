# %% imports
# libraries
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as plticker
import os

# local imports
import MNIST_dataloader
import AEarchitecture
from config_file import data_loc, batch_size
from MNIST_dataloader import *
import matplotlib.cm as cm

# set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# %% setup & data points

# create dataloader
_, test_loader, _ = MNIST_dataloader.create_dataloaders(data_loc, batch_size=1)
#Noisy_MNIST_test  = Noisy_MNIST("test" , data_loc)

# load model
model_path = os.path.join("./", "Saved_Model.pth")
model = AEarchitecture.AE()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # turn off gradients, same as model.train(False)

# load images
x_clean_test  = test_loader.dataset.Clean_Images
x_noisy_test  = test_loader.dataset.Noisy_Images
labels_test   = test_loader.dataset.Labels
# use these 10 examples as representations for all digits
x_clean_example = x_clean_test[0:10,:,:,:]
x_noisy_example = x_noisy_test[0:10,:,:,:]
labels_example = labels_test[0:10]



# forward pass through the network / INFERENCE *suspicious look*
generated,_ = model.forward(x_noisy_example)
generated_clean,_ = model.forward(x_clean_example)
generated.to('cpu')



# plot
plt.figure(figsize=(10,30))

for i in range(10):
    plt.subplot(4,10,i+1)
    plt.axis('off')
    plt.imshow(x_noisy_example[i,0,:,:].detach().numpy(), cmap='gray')
    plt.title("digit"+str(i), fontsize=10)
    plt.subplot(4,10,11+i)
    plt.axis('off')
    plt.imshow(generated[i,0,:,:].detach().numpy(), cmap='gray')
    plt.subplot(4,10,21+i)
    plt.axis('off')
    plt.imshow(x_clean_example[i,0,:,:].detach().numpy(), cmap='gray')
    plt.subplot(4,10,31+i)
    plt.axis('off')
    plt.imshow(generated_clean[i,0,:,:].detach().numpy(), cmap='gray')

plt.subplots_adjust(hspace=0.1)
plt.suptitle("Inference with noisy data")
plt.show()