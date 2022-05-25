# %% imports
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

# %%  Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()  # same as super().__init__()
        # how to find output shape: [input - kernel + 2*padding - (dilation-1)(kernel-1)]/stride + 1
        self.downsampling = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=2)  # downsamples from 32 to 28 size of img

        self.conv10 = nn.Conv2d(1, 3, kernel_size=(3,3), padding=1, stride=1, dilation=3)
        self.conv20 = nn.Conv2d(3, 2, kernel_size=(3,3), padding=1, stride=1, dilation=3)
        self.conv30 = nn.Conv2d(2, 1, kernel_size=(3,3), padding=1, stride=1)

        self.conv11 = nn.Conv2d(1, 2, kernel_size=(3,3), padding=1, stride=3)
        self.conv21 = nn.Conv2d(2, 1, kernel_size=(3,3), padding=1, stride=2)

        self.conv4 = nn.Conv2d(2, 1, kernel_size=(1,1))

        self.activ = nn.ReLU()
        self.pool2d = nn.MaxPool2d(kernel_size=(2,2), stride=2)  # new size=old size/2
        self.pool2dspecial = nn.MaxPool2d(kernel_size=(2,2), stride=1)  # new size=old size/2
        self.lastpool = nn.MaxPool2d(kernel_size=(2,1))

    def forward(self, x):
        # First branch uses dilation
        print("\n Input shape is ", x.size())
        x = self.downsampling(x)
        print("\n Input shape after downsampling is ", x.size())

        y = self.conv10(x)
        y = self.activ(y)
        y = self.pool2d(y)

        y = self.conv20(y)
        y = self.activ(y)
        y = self.pool2d(y)
        
        y = self.conv30(y)
        y = self.activ(y)
        y = self.pool2d(y)
        print("\n Encoder first branch output shape is ", y.size())

        # Second branch uses stride
        print("\n Encoder 2nd branch input shape is ", x.size())
        z = self.conv11(x)
        print("\n Encoder 2nd branch conv11 shape is ", z.size())
        z = self.activ(z)
        z = self.pool2d(z)
        print("\n Encoder 2nd branch conv11/pool2d shape is ", z.size())

        z = self.conv21(z)
        print("\n Encoder 2nd branch conv21 shape is ", z.size())
        z = self.activ(z)
        z = self.pool2dspecial(z)
        print("\n Encoder 2nd branch conv21/pool2d shape is ", z.size())
        print("\n Encoder 2nd branch output shape is ", z.size())

        # Concatenation
        h = torch.cat((y,z),dim=1)
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
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 2, kernel_size=(3,3), padding=(1,1), dilation=(2,1), stride=(1,1)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(2, 2, kernel_size=(3,3), padding=(1,1), dilation=(2,2), stride=(2,2)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(2, 2, kernel_size=(3,3), padding=(1,1), dilation=(1,1), stride=(2,2)),
            nn.ReLU()
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
        print("Setting up the autoencoder")
        print("We re feeding the autoencoder input of shape ", x.size())
        h = self.encoder(x)
        r = self.decoder(h)
        return r, h
    

# Test if the encoder works
if __name__ == "__main__":
    # imports and parameters
    from config_file import data_loc
    import MNIST_dataloader
    import autoencoder_template
    from matplotlib import pyplot as plt
    from MNIST_dataloader import create_dataloaders
    batch_size = 64
    
    # get dataloader
    train_loader, test_loader = create_dataloaders(data_loc, batch_size)
    
    # get some examples
    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)

    # try out the model before any training
    model = AE()
    output = model.forward(x_noisy_example)
    plt.figure(figsize=(12,3))
    plt.imshow(x_clean_example[0,:,:,:],cmap='gray')
    plt.show()