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
model_path = os.path.join("./", "Saved_Model.pth")
_, test_loader, _ = MNIST_dataloader.create_dataloaders(data_loc, batch_size=1)
#Noisy_MNIST_test  = Noisy_MNIST("test" , data_loc)
model = AEarchitecture.AE()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # turn off gradients, same as model.train(False)




# iterate through the test dataset # plot latent space with grid on
color_list = cm.get_cmap()#cm.rainbow(np.linspace(0, 1, 10))
plt.figure(num=1, figsize=(20,20))
ax = plt.subplot(1,1,1)
norm = colors.Normalize(vmin=0, vmax=10, clip=True)

#################DO NOT DO THIS AGAIN, REALLY SLOW###################
# for i in range(Noisy_MNIST_test.__len__()):
#     clean_img, noisy_img, label = Noisy_MNIST_test.__getitem__(i)
#     Clean_img = torch.cat((Clean_img, clean_img), 0)
#     Noisy_img = torch.cat((Noisy_img, noisy_img), 0)
#     Label = torch.cat((Label, label), 0)

i=0
for clean_img, noisy_img, label in test_loader:
    # %% get latent space
    _,latent = model.forward(clean_img)
    X = latent[0,0,0,0].to('cpu').detach().numpy()
    Y = latent[0,0,1,0].to('cpu').detach().numpy()
    label = label.to('cpu').numpy()
    plt.scatter(X, Y, c=label, cmap='tab10', norm=norm)
    i+=1
    if i == 25: break

plt.colorbar()
plt.xlabel(' $1^{st}$ coordinate')
plt.ylabel(' $2^{nd}$ coordinate')
plt.title("Latent space of test data")
ax.xaxis.set_major_locator(plticker.LinearLocator(16))
wheregridx = ax.xaxis.get_majorticklocs()
ax.yaxis.set_major_locator(plticker.LinearLocator(16))
wheregridy = ax.yaxis.get_majorticklocs()
plt.grid()
plt.show()




# %% sample data points from latent space

fig, ax = plt.subplots(15,15)
fig.set_figheight = 20
fig.set_figwidth = 20
#plt.setp(ax, xlim=(wheregridx[0], wheregridx[-1]), ylim=(wheregridy[0], wheregridy[-1]))

sample = torch.zeros([15*15,1,2,1], dtype=torch.float32).to(device)
for i in range(len(wheregridx)-1):
    for j in range(len(wheregridx)-1):
        sample[i*15+j, 0, 0, 0] = np.random.uniform(low=wheregridx[i], high=wheregridx[i+1])
        sample[i*15+j, 0, 1, 0] = np.random.uniform(low=wheregridy[i], high=wheregridy[i+1])
        img = model.decoder.forward(sample[i*15+j,:,:,:].unsqueeze(0))
        ax[i,j].imshow(img.squeeze().detach().numpy(), cmap='gray', extent=(wheregridx[i], wheregridx[i+1], wheregridy[j], wheregridy[j+1]))
        ax[i,j].axis('off')
        ax[i,j].axvspan(xmin=wheregridx[i], xmax=wheregridx[i+1], ymin=wheregridy[j], ymax=wheregridx[j+1], color='g')
plt.grid()
plt.show()

