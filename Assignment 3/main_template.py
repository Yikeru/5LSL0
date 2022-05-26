# %% imports
# libraries
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

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
model_path = os.path.join("./", "Saved_Model.pth")

# define parameters
batch_size = 64
no_epochs = 50
learning_rate = 3e-4

# get dataloader
train_loader, test_loader, val_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# create the autoencoder
AE = autoencoder_template.AE()

# set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
AE.to(device=device)

# create the optimizer
optimizer = optim.Adam(AE.parameters(), lr=learning_rate, eps=1e-08, weight_decay=0.0005)

# choose loss
criterion = nn.MSELoss(reduction='mean')


# %% training loop
print("The training will start now!!!!!")
eval_dic = {'Loss_t': [], 'train_acc': [],'Loss_v': [], 'valid_acc': []}
loss_train = []
loss_val = []

# go over all epochs
for epoch in range(no_epochs):
    print(f"\nTraining Epoch {epoch}:")
    running_loss = 0.0
    # go over all minibatches
    for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
        # fill in how to train your network using only the clean images
        AE.train()
        x_clean.to(device)
        label.to(device)
        optimizer.zero_grad()

        out, _ = AE(x_clean)
        loss = criterion(out, x_clean)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Check training loss
    train_epoch_loss = running_loss/len(train_loader)
    eval_dic['Loss_t'].append(train_epoch_loss)

    # Check validation loss
    with torch.no_grad():
        running_loss_val = 0.0
#         Set the model to evaluation mode
        AE.eval()

        for (data_clean, data_noisy, labels) in val_loader:
            # validation on noisy part or not
            x = data_clean

            # cast the inputs to the device
            x = x.to(device=device)

            output, _ = AE(x)
            loss = criterion(output, x)
            running_loss_val += loss.item()

        val_epoch_loss = running_loss_val/len(val_loader)
        eval_dic['Loss_v'].append(val_epoch_loss)

    print('Epoch', epoch)
    print('Training Loss', eval_dic['Loss_t'][epoch])
    print('Validation Loss', eval_dic['Loss_v'][epoch])

torch.save(AE.state_dict(), model_path)

# %% Plot an output after training
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

# %% HINT
#hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
x_clean_train = train_loader.dataset.Clean_Images
x_noisy_train = train_loader.dataset.Noisy_Images
labels_train  = train_loader.dataset.Labels

x_clean_test  = test_loader.dataset.Clean_Images
x_noisy_test  = test_loader.dataset.Noisy_Images
labels_test   = test_loader.dataset.Labels

# use these 10 examples as representations for all digits
x_clean_example = x_clean_test[0:10,:,:,:]
x_noisy_example = x_noisy_test[0:10,:,:,:]
labels_example = labels_test[0:10]