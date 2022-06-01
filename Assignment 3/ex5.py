# %% imports
# libraries
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from config_file import data_loc

# local imports
import MNIST_dataloader
import AEarchitecture
from config_file import data_loc
from MNIST_dataloader import *
import matplotlib.cm as cm



# %% setup & data points
model_path = os.path.join("./", "Saved_Model.pth")
train_loader, test_loader, val_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)
#Noisy_MNIST_test  = Noisy_MNIST("test" , data_loc)
model = AEarchitecture.AE()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # turn off gradients, same as model.train(False)
Clean_img=torch.Tensor([]) 
Noisy_img=torch.tensor([]) 
Label=torch.Tensor([])

# iterate through the test dataset
#################DO NOT DO THIS AGAIN, REALLY SLOW###################
# for i in range(Noisy_MNIST_test.__len__()):
#     clean_img, noisy_img, label = Noisy_MNIST_test.__getitem__(i)
#     Clean_img = torch.cat((Clean_img, clean_img), 0)
#     Noisy_img = torch.cat((Noisy_img, noisy_img), 0)
#     Label = torch.cat((Label, label), 0)
for clean_img, noisy_img, label in test_loader:
    # %% get latent space
    _,latent = model.forward(Clean_img)
    sampels = latent.size(dim=0)
    X = latent[:,0,0,0].to('cpu').detach().numpy()
    Y = latent[:,0,1,0].to('cpu').detach().numpy()
    Z = Label.to('cpu').numpy()






# plot latent space with grid on
color_list = cm.rainbow(np.linspace(0, 1, 10))
plt.figure(figsize=(20,20))
plt.figure(figsize=(20,20))

for i in range(Noisy_MNIST_test.__len__()):
    plt.annotate(Label[i].item(), (X[i][0], Y[i][1]), 
                color = color_list[Label[i]],
                fontsize= 30 )
plt.xlabel(' Horizontal')
plt.ylabel(' Vertical')
#plt.xlim((0,1))
#plt.ylim((0,1))
plt.grid()




# %%
