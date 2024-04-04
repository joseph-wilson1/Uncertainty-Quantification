### Setup data,model,ntk method

import torch
from torchvision.transforms import ToTensor
from torchvision import datasets
from tqdm import tqdm
import definitions as df
from importlib import reload
import definitions as df
import definitions_2 as df2
import torch
from tqdm import tqdm
import importlib
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import solvers
import models as model
importlib.reload(model)
importlib.reload(solvers)

torch.set_default_dtype(torch.float64)

reload(df)
reload(df2)

N_TRAIN = 1000
N_TEST = 5
N_OUTPUT = 10

training_data = datasets.MNIST(
    root="data/MNIST",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data/MNIST",
    train=False,
    download=True,
    transform=ToTensor()
)


class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def forward(self,x):
        return self.net(x)
    
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
    

training_data = torch.utils.data.Subset(training_data,range(N_TRAIN))
test_data = torch.utils.data.Subset(test_data,range(N_TEST))

batch_size = 50

train_dataloader = DataLoader(training_data, batch_size)
test_dataloader = DataLoader(test_data, N_TEST)
test_x,test_y = next(iter(test_dataloader))

ntk_net = LeNet5()
ntk_net.apply(init_weights)
ntk_net.eval()
num_weights = sum(p.numel() for p in ntk_net.parameters() if p.requires_grad)

print("Number of parameters p = {}".format(sum(p.numel() for p in ntk_net.parameters() if p.requires_grad)))
print("Number of training points = {}".format(len(training_data)))

learning_rate = 5e-3
epochs = 100

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ntk_net.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in tqdm(range(epochs)):
    cnn_train_ce = df.train_loop(train_dataloader, ntk_net, loss_fn, optimizer)
    cnn_test_ce, cnn_test_acc = df.test_loop(test_dataloader, ntk_net, loss_fn)
    if epoch % int(epochs/(epochs/4)) == 0:
        print("Epoch {} of {}".format(epoch,epochs))
        print("Training CE = {:.3f}".format(cnn_train_ce))
        print("Test CE = {:.2f}".format(cnn_test_ce))
        print("Test accuracy = {:.1f}%".format(100*cnn_test_acc))
        print("\n -------------------------------------")
print("Done!")
print("Final net training cross-entropy = {:.3f}".format(cnn_train_ce))
print("Final net test cross-entropy = {:.2f}".format(cnn_test_ce))
print("Final net test accuracy = {:.1f}%".format(100*cnn_test_acc))