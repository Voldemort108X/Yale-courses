# ps3_functions.py
# CPSC 453 -- Problem Set 3
#
# This script contains pytorch shells for a feed forward network and an autoencoder.
#
from torch._C import device
from torch.nn.functional import sigmoid, softmax, tanh
from torch import optim, nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch


class FeedForwardNet(nn.Module):
    """ Simple feed forward network with one hidden layer."""
    def __init__(self): # initialize the model
        super(FeedForwardNet, self).__init__() # call for the parent class to initialize
        # You can define variables here that apply to the entire model (e.g. weights, biases, layers...)
        # Here's how you can initialize the weight:
        # W = nn.Parameter(torch.zeros(shape)) # this creates a model parameter out of a torch tensor of the specified shape
        # ... torch.zeros is much like numpy.zeros, except optimized for backpropogation. We make it a model parameter and so it will be updated by gradient descent.

        # create a bias variable here
        self.W1 = nn.Parameter(nn.init.uniform_(torch.empty((784, 128)), a=-np.sqrt(1/128), b=np.sqrt(1/128)))
        self.b1 = nn.Parameter(nn.init.uniform_(torch.empty((1, 128)), a=-np.sqrt(1/128), b=np.sqrt(1/128)))

        self.W2 = nn.Parameter(nn.init.uniform_(torch.empty((128, 10)), a=-np.sqrt(1/10), b=np.sqrt(1/10)))
        self.b2 = nn.Parameter(nn.init.uniform_(torch.empty((1, 10)), a=-np.sqrt(1/10), b=np.sqrt(1/10)))

        # Make sure to add another weight and bias vector to represent the hidden layer.

    def forward(self, x):
        """
        this is the function that will be executed when we call the feed-fordward network on data.
        INPUT:
            x, an MNIST image represented as a tensor of shape 784
        OUTPUT:
            predictions, a tensor of shape 10. If using CrossEntropyLoss, your model
            will be trained to put the largest number in the index it believes corresponds to the correct class.
        """
        # put the logic here.
        layer1_out = torch.nn.functional.relu(torch.matmul(x, self.W1) + self.b1)
        layer2_out = softmax(torch.matmul(layer1_out, self.W2) + self.b2)

        predictions = layer2_out

        # Be sure to add some type of nonlinearity to the output of the first layer, then pass it onto the hidden layer.

        return predictions

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.enc_lin1 = nn.Linear(784, 1000)
        self.enc_lin2 = nn.Linear(1000, 500)
        self.enc_lin3 = nn.Linear(500, 250)
        self.enc_lin4 = nn.Linear(250, 2)

        self.dec_lin1 = nn.Linear(2, 250)
        self.dec_lin2 = nn.Linear(250, 500)
        self.dec_lin3 = nn.Linear(500, 1000)
        self.dec_lin4 = nn.Linear(1000, 784)
        # define additional layers here


    def encode(self, x):
        x = tanh(self.enc_lin1(x))
        x = tanh(self.enc_lin2(x))
        x = tanh(self.enc_lin3(x))
        x = self.enc_lin4(x)

        # ... additional layers, plus possible nonlinearities.
        return x

    def decode(self, z):
        # ditto, but in reverse
        z = tanh(self.dec_lin1(z))
        z = tanh(self.dec_lin2(z))
        z = tanh(self.dec_lin3(z))
        z = sigmoid(self.dec_lin4(z))

        return z

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)



# # initialize the model (adapt this to each model)
# model = FeedForwardNet()
# # initialize the optimizer, and set the learning rate
# SGD = torch.optim.SGD(model.parameters(), lr = 42000000000) # This is absurdly high.
# # initialize the loss function. You don't want to use this one, so change it accordingly
# loss_fn = torch.nn.MultiLabelSoftMarginLoss()
# batch_size = 128


def train(model, loss_fn, optimizer, train_loader, test_loader, num_epochs):
    """
    This is a standard training loop, which leaves some parts to be filled in.
    INPUT:
    :param model: an untrained pytorch model
    :param loss_fn: e.g. Cross Entropy loss of Mean Squared Error.
    :param optimizer: the model optimizer, initialized with a learning rate.
    :param training_set: The training data, in a dataloader for easy iteration.
    :param test_loader: The testing data, in a dataloader for easy iteration.
    """
    # num_epochs = 100 # obviously, this is too many. I don't know what this author was thinking.

    train_acc_list, test_acc_list = [], []

    for epoch in range(num_epochs):
        # loop through each data point in the training set
        for data, targets in train_loader:
            optimizer.zero_grad()

            # run the model on the data
            data = data.to('cuda')
            targets = targets.to('cuda')
            model_input = data.view(-1, 784)

            # TODO: Turn the 28 by 28 image tensors into a 784 dimensional tensor.
            out = model(model_input)

            # Calculate the loss
            loss = loss_fn(out,targets)

            # Find the gradients of our loss via backpropogation
            loss.backward()

            # Adjust accordingly with the optimizer
            optimizer.step()

        
        # Give status reports every 100 epochs
        if epoch % 10==0:
            train_acc, test_acc = evaluate(model,train_loader), evaluate(model,test_loader)
            train_acc_list.append(train_acc), test_acc_list.append(test_acc)
            print(f" EPOCH {epoch}. Progress: {epoch/num_epochs*100}%. ")
            print(f" Train accuracy: {train_acc}. Test accuracy: {test_acc}") #TODO: implement the evaluate function to provide performance statistics during training.

    return train_acc_list, test_acc_list

def evaluate(model, evaluation_set):
    """
    Evaluates the given model on the given dataset.
    Returns the percentage of correct classifications out of total classifications.
    """
    with torch.no_grad(): # this disables backpropogation, which makes the model run much more quickly.
        # TODO: Fill in the rest of the evaluation function.
        numOfCorrectLabels = 0

        for data, targets in evaluation_set:
            data = data.to('cuda')
            targets = targets.to('cuda')
            test_input = data.view(-1, 784)
            test_output_onehot = model(test_input)
            test_output = torch.argmax(test_output_onehot, dim=1)

            numOfCorrectLabels += (test_output == targets).float().sum()
        
        accuracy = numOfCorrectLabels / len(evaluation_set.dataset)
        
    return accuracy

# # ----- Functions for Part 5 -----
# def mmd(X,Y, kernel_fn):
#     """
#     Implementation of Maximum Mean Discrepancy.
#     :param X: An n x 1 numpy vector containing the samples from distribution 1.
#     :param Y: An n x 1 numpy vector containing the samples from distribution 2.
#     :param kernel_fn: supply the kernel function to use.
#     :return: the maximum mean discrepancy:
#     MMD(X,Y) = Expected value of k(X,X) + Expected value of k(Y,Y) - Expected value of k(X,Y)
#     where k is a kernel function
#     """

#     return mmd


# def kernel(A, B):
#     """
#     A gaussian kernel on two arrays.
#     :param A: An n x d numpy matrix containing the samples from distribution 1
#     :param B: An n x d numpy matrix containing the samples from distribution 2.
#     :return K:  An n x n numpy matrix k, in which k_{i,j} = e^{-||A_i - B_j||^2/(2*sigma^2)}
#     """

#     return K


