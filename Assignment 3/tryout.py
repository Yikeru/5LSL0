import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# local imports
import MNIST_dataloader
import autoencoder_template
from config_file import data_loc

# define parameters
batch_size = 64
no_epochs = 10
learning_rate = 3e-4

train_loader, test_loader, val_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

AE = autoencoder_template.AE()
AE.load_state_dict(torch.load(r'./Saved_Model.pth'))
AE = AE
# get some examples
examples = enumerate(test_loader)
_, (x_clean_example, x_noisy_example, labels_example) = next(examples)

example = x_clean_example[4,0,:,:]
output,_ = AE.forward(x_clean_example)

plt.figure(figsize=(12,3))
plt.subplot(1,2,1)
plt.imshow(example,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(output[0][0].detach().numpy().reshape((32,32)),cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()
