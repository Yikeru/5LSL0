# %% imports
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

# %%  Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()  # same as super().__init__()
        # how to find output shape: [input - kernel + 2*padding - (dilation-1)(kernel-1)]/stride + 1
        self.conv10 = nn.Conv2d(1, 4, kernel_size=(3,3), padding=1, stride=1, dilation=3)
        self.conv20 = nn.Conv2d(4, 3, kernel_size=(3,3), padding=1, stride=1, dilation=3)
        self.conv30 = nn.Conv2d(3, 2, kernel_size=(3,3), padding=1, stride=1)

        self.conv11 = nn.Conv2d(1, 4, kernel_size=(3,3), padding=1, stride=3)
        self.conv11 = nn.Conv2d(4, 2, kernel_size=(3,3), padding=1, stride=2)

        self.conv4 = nn.Conv2d(4, 1, kernel_size=(1,1))

        self.activ = nn.ReLU()
        self.pool2d = nn.MaxPool2d(kernel_size=(2,2), stride=2)  # new size=old size/2
        self.lastpool = nn.MaxPool2d(kernel_size=(2,1))

    def forward(self, x):
        # First branch uses dilation
        y = self.conv10(x)
        y = self.activ(y)
        y = self.pool2d(y)

        y = self.conv20(y)
        y = self.activ(y)
        y = self.pool2d(y)
        
        y = self.conv30(y)
        y = self.activ(y)
        y = self.pool2d(y)

        # Second branch uses stride
        z = self.conv11(x)
        z = self.activ(z)
        z = self.pool2d(z)

        z = self.conv21(z)
        z = self.activ(z)
        z = self.pool2d(z)

        # Concatenation
        h = torch.cat(y,z,dim=3)
        h = self.conv4(h)
        h = self.activ(h)
        h = self.lastpool(h)

        return h
    
# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # create layers here
            # nn.ConvTranspose2d(),
            # nn.ReLU(),
            # nn.Upsample(),

            # nn.ConvTranspose2d(),
            # nn.ReLU(),
            # nn.Upsample()
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
    
