# %% imports
# libraries
import torch
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# local imports
import Fast_MRI_dataloader
# import autoencoder_template
from matplotlib import pyplot as plt
from Fast_MRI_dataloader import create_dataloaders

# set-up folders
data_loc = os.getcwd()
print(data_loc)
data_loc = data_loc + '\Fast_MRI_Knee'
batch_size = 64
train_loader, test_loader = create_dataloaders(data_loc, batch_size)



class KReconstruct(nn.Module):
    def __init__(self):
        super(KReconstruct, self).__init__()
        self.model = nn.Sequential(
            # 320x320
            nn.Conv2d(2, 4, kernel_size=12, stride=2, padding=0,dilation=1),  # 155x155
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(4, 16, kernel_size=7, stride=1, padding=0, dilation=2), # 143x143
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0, dilation=1), # 139x139
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=23, stride=2, padding=0, dilation=2) # 320x320
        )

    def forward(self, x):
        xreal = torch.real(x)
        ximag = torch.imag(x)
        x = torch.cat((xreal, ximag), 1)
        return self.model(x)


model = KReconstruct()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = nn.KLDivLoss()




# Begin training for 10 epochs

epochs=10
# set device
dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Running on: ', device)
model = model.to(device=device)

print(f"\nTraining is about to start on {device}\n")
for e in range(epochs):
    print(f"\nTraining epoch {e}: \t")
    mia_loss = 0
    for batch_idx,(kspace, M, gt) in enumerate(tqdm(train_loader)):
        # prepare for training
        model.train(True)  
        optimizer.zero_grad()
        # move to gpu if available
        kspace = kspace.to(device=device)
        #M = M.to(device=device)
        gt = gt.to(device=device)
        # forward pass
        out = model(kspace.unsqueeze(1))
        # backpropagate loss
        loss = loss(out, gt)
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
        for (kspace, M, gt) in tqdm(val_loader):
            kspace = kspace.to(device=device)
            #M = M.to(device=device)
            gt = gt.to(device=device)
            out = model(kspace)
            vloss += criterion(out, gt).to(torch.float16).item()
        # update loss dictionary  
        loss_dict['val_loss'].append(vloss)
        # print validation loss
        print(f"\nEpoch validation loss is {vloss/len(val_loader)}")

# save trained model
model_path = os.getcwd() + '\Modelex5.pth'
torch.save(model.state_dict(), model_path)
    