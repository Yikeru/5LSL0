# %% imports
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

# %%  Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # create layers here
            nn.Conv2d(1, 4, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(),
            
            nn.Conv2d(1, 4, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d

    def forward(self, x):
        # use the created layers here
        h = x
        return h
    
# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # create layers here
            nn.ConvTranspose2d(),
            nn.ReLU(),
            nn.Upsample(),

            nn.ConvTranspose2d(),
            nn.ReLU(),
            nn.Upsample()
    def forward(self, h):
        # use the created layers here
        r = h
        return r
    
# %%  Autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return r, h
    
