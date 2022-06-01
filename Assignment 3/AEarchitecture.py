# %% imports
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

# %%  Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()  # same as super().__init__()
        # how to find output shape: [input - kernel + 2*padding - (dilation-1)(kernel-1)]/stride + 1
        self.encoder = nn.Sequential(
            #  
            nn.Conv2d(1,16, kernel_size=(3,3),padding=1, stride=1), #[Nx1x32x32]=> [Nx16x32x32]
            nn.ReLU(),
            nn.MaxPool2d(2,2),#[Nx16x32x32]=> [Nx16x16x16]

            nn.Conv2d(16,16, kernel_size=(3,3),padding=1, stride=1), #[Nx16x16x16]=> [Nx16x16x16]
            nn.ReLU(),
            nn.MaxPool2d(2,2),#[Nx16x16x16]=> [Nx16x8x8]

            nn.Conv2d(16,16, kernel_size=(3,3),padding=1, stride=1), #[Nx16x8x8]=> [Nx16x8x8]
            nn.ReLU(),
            nn.MaxPool2d(2,2),#[Nx16x8x8]=> [Nx16x4x4]

            nn.Conv2d(16,16, kernel_size=(3,3),padding=1, stride=1), #[Nx16x4x4]=> [Nx16x4x4]
            nn.ReLU(),
            nn.MaxPool2d(2,2),#[Nx16x4x4]=> [Nx16x2x2]

            nn.Conv2d(16,1, kernel_size=(3,3),padding=1, stride=1), #[Nx16x2x2]=> [Nx1x2x2]
            nn.MaxPool2d((1,2)) #[Nx1x2x2]=> [Nx1x2x1]
        )
    def forward(self, x):
        # First branch uses dilation
        y = self.encoder(x)
        return y
    

# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # create layers here
            # nn.ConvTranspose2d(),
            # nn.ReLU(),
            # nn.Upsample(),
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=(3,3)), # [Nx1x2x1]=>[Nx16x4x3]
            nn.ReLU(),
            nn.Upsample(scale_factor=2), #[Nx16x4x3]=>[Nx16x8x6]

            nn.ConvTranspose2d(16, 16, kernel_size=(3,3)), # [Nx16x8x6]=>[Nx16x10x8]
            nn.ReLU(),
            nn.Upsample(scale_factor=2), #[Nx16x10x8]=>[Nx16x20x16]

            nn.ConvTranspose2d(16, 16, kernel_size=(3,3)), # [Nx16x10x8]=>[Nx16x22x18]
            nn.ReLU(),
            nn.Upsample(scale_factor=2), #[Nx16x22x18]=>[Nx16x44x36]

            nn.ConvTranspose2d(16, 1, kernel_size=(3,3)), # [Nx16x44x36]=>[Nx1x46x38]
            nn.Upsample(size=(32,32)), #[Nx1x46x38]=>[Nx1x32x32]
            )

    def forward(self, h):
        # use the created layers here
        return self.decoder(h)


# %%  Autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
#         print("Setting up the autoencoder")
#         print("We re feeding the autoencoder input of shape ", x.size())
        h = self.encoder(x)
        r = self.decoder(h)
        return r, h