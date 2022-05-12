import torch.nn as nn
import numpy as np
import os

from MNIST_dataloader import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current file marks the root directory
MODEL_NEW = os.path.join(ROOT_DIR, "Newest_model")    # Directory for storing the newest model
MODEL_BEST = os.path.join(ROOT_DIR, 'best_model')     # Directory for storing the best model


class FFNN_1b(nn.Module):
    """
    A simple Feed Forward Neural Network
    
    Param
    ------------
    input_size : int (default = 786)         
        The length of the datasets which is width*height)
    n_hidden : int (default: 200)
        Number of hidden units.
    n_classes: int  (default = 10)        
        Number of the final classes, in our case 0-9 thus 10 classes.
    """
    def __init__(self, input_size=786, hidden_size= 200, n_classes=10):
        super().__init__()
        
        self.flat = nn.Flatten(),
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size),

        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, n_classes)

        #no activation functions yet
            
    
    def forward(self, x):
        # forward always defines connectivity
        y = self.network(x)
        return y
    
    def predict(self,x):
        """Predict class labels
        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        Returns:
        ----------
        y_pred : array, shape = [n_samples]
            Predicted class labels.
        """
        # Implement prediction here
        a_out = self.forward(x).detach().numpy()
        result = np.zeros(len(a_out))
        for i in range(len(a_out)):
            j = np.argmax(a_out[i])
            result[i] = j
        y_pred = result
        return y_pred
    
model = FFNN_1b()

def train_part34(model, optimizer, epochs=1, write_to_file=False, USE_GPU=True):
    """
    Fit the model on the training data set.
    Arguments
    ---------
    model : model class
        Model structure to fit, as defined by build_model().
    epochs :  int
        Number of epochs the training should do
    optimizer : optim class
        Optimizer to use
    write_to_file : bool (default:False)
        Write model to file; can later be loaded through load_model()
    USE_GPU : bool (default:True
        Parameters that when set to True uses the GPU for the training
    Returns
    -------
    model : model class
        The trained model.
    """
    # First check if we have GPU we can use to train
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print_every = 100
    
    
    
    model = model.to(device=device)  # Move the model parameters to CPU/GPU
    print_every = 100 # Constant to control how frequently we print train loss
    
    # Define the trainlaoder
#     X_train, y_train, X_val, y_val = load_data_set(TRAINING_IMAGE_DIR, n_validation=0.2)
#     trainloader =  custom_dataloader(X_train, y_train, bs = 20, Shuffle= True)
#     valloader = custom_dataloader(X_val, y_val, bs = 20, Shuffle= True)
    
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()

    #### Save the model
    print(acc)
    if write_to_file:        
        # Save the weights of the model to a .pt file
        print(MODEL_NEW)
        torch.save(model.state_dict(), MODEL_NEW)

    return model