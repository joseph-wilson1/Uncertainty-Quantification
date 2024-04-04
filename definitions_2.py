import torch
import importlib
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import datetime
import os
import solvers as solv
import definitions as df
from importlib import reload

reload(solv)
reload(df)

from scipy.sparse.linalg import LinearOperator, lsmr
from tqdm import tqdm
from torch.func import vmap, jacrev
from functorch import make_functional, make_functional_with_buffers

torch.set_default_dtype(torch.float64)

def train_loop(dataloader, model, loss_fn, optimizer, train_mode=True):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    if train_mode:
        model.train()
    else:
        model.eval()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        total_loss += loss

    total_loss_mean = total_loss / (batch + 1)
    return total_loss_mean


def test_loop(dataloader, model, loss_fn, verbose=False):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    
    return test_loss, correct

def training(train_loader, test_loader, model, loss_fn, optimizer, epochs: int = 50, verbose: bool = False):
    '''
    Training function. Will train and test, and will report metrics.

    Outputs:
        - train_ce
        - test_ce
        - test_acc
    '''

    for epoch in tqdm(range(epochs)):
        train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        test_loss, test_acc = test_loop(test_loader, model, loss_fn)
        if verbose:
            if epoch % int(epochs/(epochs/4)) == 0:
                print("Epoch {} of {}".format(epoch,epochs))
                print("Training CE = {:.4f}".format(train_loss))
                print("Test CE = {:.4f}".format(test_loss))
                print("Test accuracy = {:.1f}%".format(100*test_acc))
                print("\n -------------------------------------")
    if verbose:
        print("Done!")
        print("Final training cross-entropy = {:.4f}".format(train_loss))
        print("Final test cross-entropy = {:.4f}".format(test_loss))
        print("Final test accuracy = {:.1f}%".format(100*test_acc))
    return train_loss, test_loss, test_acc