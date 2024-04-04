### --- Dependencies --- ###
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
from importlib import reload

reload(solv)

importlib.reload(solv)
from scipy.sparse.linalg import LinearOperator, lsmr
from tqdm import tqdm
from torch.func import vmap, jacrev
from functorch import make_functional, make_functional_with_buffers

torch.set_default_dtype(torch.float64)

class RegressionDataset(Dataset):
    '''
  Prepare dataset for regression.
  Input the number of features.

  Input:
   - dataset: numpy array

   Returns:
    - Tuple (X,y) - X is a numpy array, y is a double value.
  '''
    def __init__(self, dataset, input_dim, mX=0, sX=1, my=0, sy=1):
        self.X, self.y = dataset[:,:input_dim], dataset[:,input_dim]
        self.X, self.y = (self.X - mX)/sX, (self.y - my)/sy
        self.len_data = self.X.shape[0]
        self.X, self.y = torch.from_numpy(self.X), torch.from_numpy(self.y)

    def __len__(self):
        return self.len_data

    def __getitem__(self, i):
        return self.X[i,:], self.y[i]
    
def data_split(df, train_ratio):
    num_points = len(df)
    train_size = int(num_points*train_ratio)
    dataset_numpy = df.values
    np.random.shuffle(dataset_numpy)
    training_set, test_set = dataset_numpy[:train_size,:], dataset_numpy[train_size:,:]
    print("training set has shape {} \n".format(training_set.shape))
    print("test set has shape {}".format(test_set.shape))
    return training_set, test_set

class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self, input_d, width):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(input_d, width),
      nn.ReLU(),
      nn.Linear(width,width),
      nn.ReLU(),
      nn.Linear(width,width),
      nn.ReLU(),
      nn.Linear(width,width),
      nn.ReLU(),
      nn.Linear(width,width),
      nn.ReLU(),
      nn.Linear(width, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)
  
class EnsembleNetwork(nn.Module):
    def __init__(self, input_d, width):
        super().__init__()
        self.linear_1 = nn.Linear(input_d,width)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(width,width)
        self.linear_mu = nn.Linear(width,1)
        self.linear_sig = nn.Linear(width,1)

    def forward(self, x):
        x = self.relu(self.linear_1(x)) #Relu applied to input layer
        x = self.relu(self.linear_2(x)) #Relu applied to first hidden layer
        x = self.relu(self.linear_2(x)) #Relu applied to second hidden layer
        x = self.relu(self.linear_2(x)) #Relu applied to third hidden layer
        x = self.relu(self.linear_2(x)) #Relu applied to fourth hidden layer
        mu = self.linear_mu(x)
        variance = self.linear_sig(x)
        variance = F.softplus(variance) + 1e-6
        return mu, variance

def to_np(x):
    return x.cpu().detach().numpy()

class CustomNLL(nn.Module):
    def __init__(self):
        super(CustomNLL, self).__init__()

    def forward(self, y, mean, var):
        
        loss = (0.5*torch.log(var) + 0.5*(y - mean).pow(2)/var).mean() + 1

        if np.any(np.isnan(to_np(loss))):
            print(torch.log(var))
            print((y - mean).pow(2)/var)
            raise ValueError('There is Nan in loss')
        
        return loss
  
def weights_init(m):
    p = sum(p.numel() for p in m.parameters() if p.requires_grad)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # nn.init.normal_(m.weight,mean=0,std=1/p)
        nn.init.normal_(m.bias,mean=0,std=0.01)
        # nn.init.xavier_normal_(m.bias)

def optimizer_shared(model, type='adam', learning_rate=1e-1):
    if type=='adam':
        return torch.optim.Adam(model.parameters(), lr = learning_rate)
    elif type=='sgd':
        return torch.optim.SGD(model.parameters(), lr = learning_rate)
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def training_loop_ntk(dataloader, model, optimizer, loss_function, verbose=False):
    model.train()
    train_loss_total = 0
    for i, (X,y) in enumerate(dataloader):
        # Get and prepare inputs
        y = y.reshape((y.shape[0],1))

        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        pred = model(X)
        
        # Compute loss
        loss = loss_function(pred, y)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        train_loss = loss.item()

        train_loss_total += train_loss
        # Print statistics
        if verbose:
            print("train loss for batch {} is {}".format(i+1, train_loss))

    train_loss_mean = train_loss_total / i
    return train_loss_mean

def test_loop_ntk(dataloader, model, loss_function, my = 0, sy = 1, verbose=False):
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            y = y.reshape((y.shape[0],1))
            pred = model(X)
            pred = pred * sy + my
            test_loss = loss_function(pred, y).item()
            rel_error = torch.square(y-pred)
            rmse = (loss_function(pred, y).item())**0.5
            if verbose:
                print("--- mse = {:.2f} ---".format(test_loss))
    return test_loss, rel_error

def training_loop_ensemble(dataloader, model, optimizer, loss_function, mse_loss, verbose=False):
    model.train()
    train_loss_total = 0
    for i, (X,y) in enumerate(dataloader):
        
        # Get and prepare inputs
        y = y.reshape((y.shape[0],1))
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        mean, variance = model(X)
        
        # Compute loss
        loss = loss_function(y, mean, variance)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        train_loss = loss.item()

        train_loss_total += mse_loss(y,mean).item()

        # Print statistics
        if verbose:
            print("train loss for batch {} is {}".format(i+1, train_loss))

    train_loss_mean = train_loss_total / i
    return train_loss_mean

def test_loop_ensemble(dataloader, model, my, sy, loss_function, mse_loss, verbose=False):
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            y = y.reshape((y.shape[0],1))
            mean, variance = model(X)
            mean = mean * sy + my
            variance = variance * (sy**2)
            test_loss = loss_function(y, mean, variance)
            rel_error = torch.square(y-mean)
            if verbose:
                print("--- Test NLL = {:.2f} ---".format(test_loss))
    mse = mse_loss(y,mean).item()
    RMSE = mse**0.5
    return mse, test_loss, rel_error

def flatten_extend_gradient(parameters):
    flat_list = []
    for parameter in parameters:
        flat_list.extend(parameter.grad.detach().numpy().flatten())
    return flat_list

def gradient_model(model,optimizer,xi):
    ## model needs to have parameters with requires_grad=true
    optimizer.zero_grad()
    model(xi).backward()
    grad_vec = np.array(flatten_extend_gradient(list(model.parameters())))
    return grad_vec

def ntk_single(x1,x2,model, optimizer):
    j1 = gradient_model(model=model,optimizer=optimizer,xi=x1)
    j2 = gradient_model(model=model,optimizer=optimizer,xi=x2)
    return j1 @ j2.transpose()

def ntk_matrix(X1,X2,model,optimizer):
    # Xi must be a torch variable
    Kappa = np.empty((len(X1),len(X2)))
    with tqdm(total=int(len(X1)*len(X2))) as pbar:
        for i1,x1 in enumerate(X1):
            if type(x1) is tuple:
                x1,_ = x1
            for i2,x2 in enumerate(X2):
                if type(x2) is tuple:
                    x2,_ = x2
                Kappa[i1,i2] = ntk_single(x1,x2,model,optimizer)
                pbar.update(1)
    return Kappa

def MVP_JTX(v,model,X_training, optimizer):
    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mvp = np.zeros((p,1))
    for i,(xi,_) in enumerate(X_training):
        xi = torch.from_numpy(xi)
        g = gradient_model(model=model,optimizer=optimizer,xi=xi).reshape((p,1))
        mvp += v[i]*g
    return mvp

def MVP_JX(v,model,X_training, optimizer):
    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n = len(X_training)
    mvp = np.zeros((n,1))
    v = v.reshape((p,1))
    for i,(xi,_) in enumerate(X_training):
        xi = torch.from_numpy(xi)
        g = gradient_model(model=model,optimizer=optimizer,xi=xi).reshape((p,1))
        mvp[i,0] = g.transpose() @ v
    return mvp

def MVP_JJT(v,model,X_training, optimizer):
    x1 = MVP_JTX(v,model,X_training, optimizer)
    x2 = MVP_JX(x1,model,X_training, optimizer)
    return x2

def ntk_uncertainty_explicit(Kappa, train_dataset, test_dataset, model, optimizer, type='iterative', rtol=1e-9, maxit=50):
    uncertainty_array = np.empty((1,len(test_dataset)))
    solver_info = np.empty((3,len(test_dataset)))
    with tqdm(total=int(len(test_dataset))) as pbar:
        for i,(x,_) in enumerate(test_dataset):
            x = x.reshape((1,8))
            kappa_xx = ntk_matrix(x, x, model, optimizer)
            kappa_xX = ntk_matrix(x, train_dataset, model, optimizer)
            if type=='direct':
                uncertainty_estimate = kappa_xx - kappa_xX @ np.linalg.solve(Kappa,kappa_xX.transpose())
            elif type=='iterative':
                x_solve, it, resid, rel_resid, rel_mat_resid = solv.CR(Kappa,kappa_xX.transpose(),rtol=rtol,init=False, maxit=maxit, VERBOSE=False)
                x_solve = lifted_solution(x_solve,resid)
                uncertainty_estimate = kappa_xx - kappa_xX @ x_solve
                solver_info[0,i] = it
                solver_info[1,i] = rel_resid
                solver_info[2,i] = rel_mat_resid
            uncertainty_array[0,i] = uncertainty_estimate
            pbar.update(1)
    return uncertainty_array, solver_info

def ensemble_result(test_loader, ensemble_M, model_list, sy=1, my=0):
    mu_test_list = np.empty((ensemble_M,len(test_loader.dataset)))
    sigma_test_list = np.empty((ensemble_M,len(test_loader.dataset)))
    for i in range(ensemble_M):
        for X,y in test_loader:
            y = y.reshape((y.shape[0],1))
            mu, sig = model_list[i](X)
            mu = mu * sy + my
            sig = sig * (sy**2)
            mu_test_list[i,:] = np.reshape(to_np(mu), (len(test_loader.dataset)))
            sigma_test_list[i,:] = np.reshape(to_np(sig),(len(test_loader.dataset)))
    mu_mean = np.mean(mu_test_list,axis=0)
    sigma_mean = np.mean(sigma_test_list, axis=0) + np.mean(np.square(mu_test_list), axis = 0) - np.square(mu_mean)
    return mu_mean, sigma_mean

def calibration_curve_ntk(testloader, uncertainties, model, num_c,my=0,sy=1):
    c = np.linspace(0,1,num_c)
    observed_true = np.empty(num_c)
    total = uncertainties.size
    for i, (X,y) in enumerate(testloader):
         Yhat_pre = model(X)
         Yhat = Yhat_pre.detach().numpy()*sy + my
         y = y.detach().numpy()
    for i,ci in enumerate(c):
        z = scipy.stats.norm.ppf((1+ci)/2)
        ci_c = z * np.sqrt(uncertainties*(sy**2))
        left_ci = y >= (Yhat - ci_c.reshape(-1,1)).squeeze(1)
        right_ci = y <= (Yhat + ci_c.reshape(-1,1)).squeeze(1)
        observed_true_c = np.logical_and(left_ci,right_ci)
        num_true = observed_true_c[observed_true_c==True].size
        observed_true[i] = num_true/total
        # print(num_true)
    return observed_true

def calibration_curve_ensemble(testloader, mu, sigma2, num_c):
    c = np.linspace(0,1,num_c)
    observed_true = np.empty(num_c)
    total = mu.size
    for i, (_,y) in enumerate(testloader):
         y = y.detach().numpy()
    for i,ci in enumerate(c):
        z = scipy.stats.norm.ppf((1+ci)/2)
        ci_c = z * np.sqrt(sigma2)
        left_ci = y >= (mu - ci_c)
        right_ci = y <= (mu + ci_c)
        observed_true_c = np.logical_and(left_ci,right_ci)
        num_true = observed_true_c[observed_true_c==True].size
        observed_true[i] = num_true/total
        # print(num_true)
    return observed_true

def plot_calibration(observed_true_ntk, observed_true_ensemble, dataset_str, dir_name, plot_name):
    num_c = observed_true_ntk.size
    c = c = np.linspace(0,1,num_c)
    plt.plot(c,c)
    plt.plot(c,observed_true_ntk, label='NTK')
    plt.plot(c,observed_true_ensemble, label='Deep Ensemble')
    plt.xlabel("Expected accuracy")
    plt.ylabel("Observed accuracy")
    plt.title("Calibration curve".format(dataset_str))
    plt.legend()
    plt.savefig(dir_name + plot_name, format="pdf", bbox_inches="tight")
    plt.show()

### Classification definitions
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 3, 5)
        self.fc1 = nn.Linear(3 * 4 * 4, 10)
        # self.fc2 = nn.Linear(120,84)
        # self.fc3 = nn.Linear(84,10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x,len(x.shape)-3)
        x = self.fc1(x)
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x    

def train_loop(dataloader, model, loss_fn, optimizer, verbose=False, train_mode=True):
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

        if verbose:
            current = (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
    if verbose:
        print(f"Test Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, correct

def flatten_extend_gradient(parameters):
    flat_list = []
    for parameter in parameters:
        flat_list.extend(parameter.grad.detach().numpy().flatten())
    return flat_list

def gradient_model_c(model,xi,optimizer,c):
    ## model needs to have parameters with requires_grad=true
    optimizer.zero_grad()
    model(xi)[c].backward()
    grad_vec = np.array(flatten_extend_gradient(list(model.parameters())))
    return grad_vec

def ntk_single_matrix(x1,x2,model,optimizer,num_class,p):
    ## Returns matrix size c x c
    gradf_x1 = np.empty((num_class,p))
    gradf_x2 = np.empty((num_class,p))
    for c in range(num_class):
        gradf_x1[c,:] = gradient_model_c(model,x1,optimizer,c)
        gradf_x2[c,:] = gradient_model_c(model,x2,optimizer,c)
    j = gradf_x1 @ gradf_x2.transpose()
    return j 

def ntk_single_c(x1,x2,model,optimizer,c):
    ## Returns scalar
    j1 = gradient_model_c(model,x1,optimizer,c)
    j2 = gradient_model_c(model,x2,optimizer,c)
    j = j1 @ j2.transpose()
    return j

def ntk_matrix_allc(X1,X2,model,optimizer,num_class):
    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n1 = len(X1)
    n2 = len(X2)
    Kappa = np.empty((n1*num_class,n2*num_class))
    nidx1 = 0
    nidx2 = 0
    with tqdm(total=int(n1*n2)) as pbar:
        for i1,x1 in enumerate(X1):
            if type(x1) is tuple:
                x1,_ = x1
            x1 = x1.reshape((1,28,-1))
            for i2,x2 in enumerate(X2):
                if type(x2) is tuple:
                    x2,_ = x2
                x2 = x2.reshape((1,28,-1))
                J = ntk_single_matrix(x1,x2,model,optimizer,num_class,p)
                incx1 = J.shape[0]
                incx2 = J.shape[1]
                Kappa[nidx1:nidx1+incx1,nidx2:nidx2+incx2] = J
                nidx2 += incx2
                pbar.update(1)
            nidx1 += incx1
            nidx2 = 0
    return Kappa

def ntk_matrix_c(X1,X2,model,optimizer,c):
# x1, x2 must become torch variables
    n1 = len(X1)
    n2 = len(X2)
    Kappa = np.empty((n1,n2))
    with tqdm(total=int(len(X1)*len(X2))) as pbar:
        for i1,x1 in enumerate(X1):
            print(i1)
            if type(x1) is tuple:
                x1,_ = x1
            x1 = x1.reshape((1,28,-1))
            for i2,x2 in enumerate(X2):
                if type(x2) is tuple:
                    x2,_ = x2
                x2 = x2.reshape((1,28,-1))
                Kappa[i1,i2] = ntk_single_c(x1,x2,model,optimizer,c)
                pbar.update(1)
    return Kappa

def MVP_JTX_c(v,model,X_training,optimizer,c):
    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mvp = np.zeros((p,1))
    for i,(xi,_) in enumerate(X_training):
        g = gradient_model_c(model,xi,optimizer,c).reshape((p,1))
        mvp += v[i]*g
    return mvp

def MVP_JX_c(v,model,X_training,optimizer,c):
    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n = len(X_training)
    mvp = np.zeros((n,1))
    v = v.reshape((p,1))
    for i,(xi,_) in enumerate(X_training):
        g = gradient_model_c(model,xi,optimizer,c).reshape((p,1))
        mvp[i,0] = g.transpose() @ v
    return mvp

def MVP_JJT_c(v,model,X_training,optimizer,c):
    x1 = MVP_JTX_c(v,model,X_training,optimizer,c)
    x2 = MVP_JX_c(x1,model,X_training,optimizer,c)
    return x2

def ntk_uncertainty_classification(train_dataset, test_dataset, model, optimizer, num_c):
    with tqdm(total=len(test_dataset)*num_c) as pbar:
        for i,(xi,yi) in enumerate(test_dataset):
            for c in range(num_c):
                kappa_xX = ntk_matrix_c(train_dataset,xi,model,optimizer,c)
                mvp = lambda v : MVP_JJT_c(v=v,model=model,X_training=train_dataset,optimizer=optimizer,c = c)
                A = LinearOperator((len(train_dataset),len(train_dataset)), matvec=mvp)
                b = kappa_xX
                x,_,_,_,_ = solv.CR(A,b,rtol=1e-7,maxit=50,VERBOSE=False)
                kappa_xx = ntk_single_c(xi,xi,model,optimizer,c)
                uq = kappa_xx - b.transpose() @ x
                pbar.update(1)

def ntk_uncertainty_classification_single(train_dataset, test_point_x, model, optimizer, num_c, mode='max'):
    if mode=='max':
        c = torch.argmax(model(test_point_x)).item()
        classes = [c]
    elif mode=='all':
        classes = range(num_c)
    uncertainty_array = np.empty((1,len(classes)))
    with tqdm(len(classes)) as pbar:
        for i,c in enumerate(classes):
            kappa_xX = ntk_matrix_c(train_dataset,test_point_x,model,optimizer,c)
            mvp = lambda v : MVP_JJT_c(v=v,model=model,X_training=train_dataset,optimizer=optimizer,c = c)
            A = LinearOperator((len(train_dataset),len(train_dataset)), matvec=mvp)
            b = kappa_xX
            x,_,_,_,_ = solv.CR(A,b,rtol=1e-7,maxit=50,VERBOSE=False)
            kappa_xx = ntk_single_c(test_point_x,test_point_x,model,optimizer,c)
            uq = kappa_xx - b.transpose() @ x
            uncertainty_array[0,i] = uq
            pbar.update(1)
    return uncertainty_array
        
def general_probit_approx(mu,sigma):
    return mu/(np.sqrt(1+np.pi/8*sigma))

def ntk_uncertainty_explicit(Kappa, train_dataset, test_dataset, model, type='direct', rtol=1e-9, maxit=50, epsilon=0):
    fnet, params = make_functional(model)     

    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0) 
    
    uncertainty_array = np.empty((1,len(test_dataset)))
    uncertainty_array_lift = np.empty((1,len(test_dataset)))
    solver_info = np.empty((3,len(test_dataset)))
    test_dataloader = DataLoader(test_dataset,1)
    train_dataloader = DataLoader(train_dataset,len(train_dataset))
    X_train,y_train = next(iter(train_dataloader))

    x_hat = model(X_train)
    resid_array = torch.reshape(y_train,(-1,1)) - x_hat
    print("Norm of residual array (y-f) = {}".format(torch.norm(resid_array)))
    resid_solve,_,_,_,_ = solv.CR(
        Kappa,
        resid_array.detach().numpy(),
        rtol=rtol,
        init=False,
        maxit=maxit,
        VERBOSE=False
    )
    mu = np.empty((1,len(test_dataset)))
    print("Norm of resid_solve = {}".format(np.linalg.norm(resid_solve)))

    if type=='direct':
        s = Kappa.shape
        Kappa = Kappa + epsilon*np.eye(s[0])


    with tqdm(total=int(len(test_dataset))) as pbar:
        for i,(x_test,_) in enumerate(test_dataloader):
            kappa_xx = empirical_ntk_jacobian_contraction(
                fnet_single=fnet_single,
                params=params,
                x1=x_test,
                x2=x_test
                ).detach().numpy().squeeze((2,3))
            kappa_xX = empirical_ntk_jacobian_contraction(
                fnet_single=fnet_single,
                params=params,
                x1=x_test,
                x2=X_train
                ).detach().numpy().squeeze((2,3))
            if type=='direct':
                uncertainty_estimate = kappa_xx - kappa_xX @ np.linalg.solve(Kappa,kappa_xX.transpose())
            elif type=='iterative':
                x_solve, it, resid, rel_resid, rel_mat_resid = solv.CR(
                    Kappa,
                    kappa_xX.transpose(),
                    rtol=rtol,
                    init=False, 
                    maxit=maxit, 
                    VERBOSE=False
                    )
                kappa_hat = lifted_solution(x_solve,resid)
                uncertainty_estimate = kappa_xx - kappa_xX @ x_solve
                lifted_ue = kappa_xx - kappa_hat.transpose() @ Kappa @ kappa_hat
                solver_info[0,i] = it
                solver_info[1,i] = rel_resid
                solver_info[2,i] = rel_mat_resid
                uncertainty_array_lift[0,i] = lifted_ue
            uncertainty_array[0,i] = uncertainty_estimate

            mu[0,i] = kappa_xX @ resid_solve + x_hat[i].detach().numpy()
            pbar.update(1)

    
    return uncertainty_array, solver_info, uncertainty_array_lift, mu

def ntk_uncertainty_single_class(Kappa,train_dataset,test_x,model,c,rtol,maxit):
    '''
    Inputs:
         - fnet: is single functional model output.
         - Kappa: is n x n x c x c ntk matrix, of numpy dtype.
         - train_dataset: is torch.tensor of training input data, n x (img x img).
         - test_x: is torch.tensor of test input point, (img x img).
         - c: is class.
    '''
    fnet, params = make_functional(model)     
    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)
    
    print("Inside loop shape is {}".format(test_x.shape))
    
    kappa_xx = empirical_ntk_jacobian_contraction_c(
                    fnet_single=fnet_single,
                    params=params,
                    x1=test_x,
                    x2=test_x,
                    c = c
                    )
    kappa_xX = empirical_ntk_jacobian_contraction_c(
                    fnet_single=fnet_single,
                    params=params,
                    x1=test_x,
                    x2=train_dataset,
                    c = c
                    )
    
    print("Before solver kappa {}".format(Kappa[:,:,c,c].shape))
    print("Before solver kappa_xX {}".format(kappa_xX.transpose(0,1).shape))

    x_solve, _, resid, _, _ = solv.CR_torch(
                        Kappa[:,:,c,c],
                        kappa_xX.transpose(0,1).squeeze(1),
                        rtol=rtol,
                        init=False, 
                        maxit=maxit, 
                        VERBOSE=False
                        )
    
    x_solve = lifted_solution(x_solve,resid)
    sigma2 = kappa_xx - kappa_xX @ x_solve
    return sigma2


def ntk_method(train, test, model, num_class: int = 10, solver: str = 'direct_direct', batch_size: int = 0, softmax: bool = False, cr_maxit: int = 100, cr_rtol: float = 1e-12, lsmr_maxit: int = 30):
    '''
    Calculates mu,sigma2 for all points in test, for model trained on train set.

    solver options:
     - 'direct_direct'
     - 'direct_iterative'
     - 'iterative_iterative_cr'
     - 'iterative_iterative_lsmr'
    '''
    ### Model info/fnet
    fnet, params, buffers = make_functional_with_buffers(model)
    num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ### Set up storage
    sigma2 = torch.empty((len(test),num_class))
    mu = torch.empty((len(test),num_class))

    ### Datasets and DataLoaders
    test_loader = DataLoader(test,1)
    train_loader = DataLoader(train,len(train))
    train_loader_individual = DataLoader(train,1)
    train_x,train_y = next(iter(train_loader))
    
    ### Residual (y-f) for mu
    train_y = one_hot(train_y,num_classes=num_class)
    if softmax:
        train_y_hat = model(train_x).softmax(dim=1)
    else:
        train_y_hat = model(train_x)
    train_residual = train_y - train_y_hat

    with tqdm(total=int(len(test)*num_class)) as pbar:
        ### Directly form Kappa
        if solver == 'direct_direct' or solver == 'direct_iterative':
            for c in range(num_class):
                ### Take class c from output of model
                def fnet_single(params, x):
                    return fnet(params, buffers, x.unsqueeze(0)).squeeze(0)[c].reshape(1)
                
                ### Form kappa
                Kappa = empirical_ntk(fnet_single,params,train,train,batch_size).detach()

                for i,(x_t,_) in enumerate(test_loader):
                    ### Form small kappas
                    kappa_xX = empirical_ntk(fnet_single,params,train,x_t,batch_size).detach()
                    kappa_xx = empirical_ntk(fnet_single,params,x_t,x_t,batch_size).detach()

                    ### Directly solve K^-1
                    if solver == 'direct_direct':
                        Kappa_solve = torch.linalg.solve(Kappa, kappa_xX)

                    ### Iteratively solve K^-1 kappa_xX
                    elif solver == 'direct_iterative':
                        Kappa_solve,_,_,_,_ = solv.CR_torch(Kappa,kappa_xX.reshape(-1),rtol=1e-12,VERBOSE=False)

                    ### Uncertainty value
                    sigma2_i = kappa_xx - kappa_xX.reshape(1,-1) @ Kappa_solve.reshape(-1,1)
                    sigma2[i,c] = sigma2_i

                    ### Mean value
                    mu_i = Kappa_solve.reshape(1,-1) @ train_residual[:,c].reshape(-1,1) + model(x_t).squeeze(0)[c]
                    mu[i,c] = mu_i

                    ### Update progress bar
                    pbar.update(1)
        
        ### Don't directly form Kappa - use MVP
        if solver == 'iterative_iterative_cr' or solver == 'iterative_iterative_lsmr':
            for c in range(num_class):
                ### Take class c from output of model
                def fnet_single(params, x):
                    return fnet(params, buffers, x.unsqueeze(0)).squeeze(0)[c].reshape(1)
                
                ### Define MVPs for fnet(c)
                Ax = lambda x: JTw(x,fnet_single,params,train_loader_individual)
                ATx = lambda x: Jw(x,fnet_single,params,train_loader_individual)
                Kappa = lambda x: ATx(Ax(x))

                ### LinearOperator for use with scipy.linalg.lsmr
                A = LinearOperator((num_weights,len(train)),matvec=Ax,rmatvec=ATx)

                for i,(x_t,_) in enumerate(test_loader):
                    ### Form small kappas
                    kappa_xx = empirical_ntk(fnet_single,params,x_t,x_t,batch_size).detach()
                    kappa_xX = ATx(grad_f(x_t,fnet_single,params)) # we form this way as using empirical_ntk takes 8 * n * p bytes.

                    ### Iteratively solve K^-1 kappa_xX using MVP and lsmr
                    if solver == 'iterative_iterative_lsmr':
                        ### b = grad_f(test_point)
                        b = grad_f(x_t,fnet_single,params).cpu().detach().numpy() #lsmr is a numpy function, must cast to numpy
                        Kappa_solve = lsmr(A,b,show=False,maxiter=lsmr_maxit)
                        Kappa_solve = torch.from_numpy(Kappa_solve[0]).detach()

                    ### Iteratively solve K^-1 kappa_xX using MVP and CR
                    elif solver == 'iterative_iterative_cr':
                        Kappa_solve = solv.CR_torch(Kappa,kappa_xX.reshape(-1),rtol=cr_rtol,maxit=cr_maxit,VERBOSE=True)
                        Kappa_solve = Kappa_solve[0]

                    ### Uncertainty value
                    sigma2_i = kappa_xx - kappa_xX.reshape(1,-1) @ Kappa_solve.reshape(-1,1)
                    sigma2[i,c] = sigma2_i

                    ### Mean value
                    mu_i = Kappa_solve.reshape(1,-1) @ train_residual[:,c].reshape(-1,1) + model(x_t).squeeze(0)[c]
                    mu[i,c] = mu_i

                    ### Update progress bar
                    pbar.update(1)

    return mu, sigma2

def one_hot(y,num_classes):
        yh = torch.zeros((y.shape[0],num_classes),dtype=torch.float64)
        yh[torch.arange(y.shape[0]),y] = 1
        return yh

def grad_f(x,fnet_single,params):
    '''
    Function to calculate gradient of net, evaluated at x, and reshape into list of size p x 1.

    Input:
        - x: list of torch.tensor.

    OUTPUT:
        - gf: gradient vector size (p)
    '''
    gf = torch.cat([gfi.reshape(-1) for gfi in jacrev(fnet_single)(params,x)]).detach()
    return gf

def JTw(x,fnet_single,params,training_loader):
    '''
    Function to calculate the MVP J^T(training_data,net) w, where J is len(training_data) x p, and w is n x 1 (p).

    Input:
        - x: numpy_array or torch.tensor, size len(training_data)
        - fnet_single
        - params
        - training_loader: DataLoader of training data, batch_size=1
    '''
    m = type(x).__module__
    if m == np.__name__:
        x = torch.from_numpy(x)
    jtw = 0
    for i,(y,_) in enumerate(training_loader):
        jtw += grad_f(y,fnet_single,params)*x[i]
    if m == np.__name__:
        jtw = jtw.cpu().detach().numpy()
    return jtw

def Jw(x,fnet_single,params,training_loader):
    '''
    Function to calculate the MVP J(training_data,net) w, where J is len(training_data) x p, and w is p x 1 (p).

    Input:
        - x: numpy_array or torch.tensor, size p (num_params)
        - fnet_single
        - params
        - training_loader: DataLoader of training data, batch_size=1
    '''
    m = type(x).__module__
    if m == np.__name__:
        x = torch.from_numpy(x)
    gfw = lambda y : torch.dot(grad_f(y,fnet_single,params),x)
    jw = torch.empty((len(training_loader)))
    for i,(y,_) in enumerate(training_loader):
        jw[i] = gfw(y)
    if m == np.__name__:
        jw = jw.cpu().detach().numpy()
    return jw

def empirical_ntk(fnet_single, params, x1, x2, batch_size=0):
    '''
    INPUT:
        - fnet_single: must be function form of single-output NN
        - params:
        - x1: either a torch.dataset/torch.subset type or a list containing a single torch.tensor
        - x2: either a torch.dataset/torch.subset type or a list containing a single torch.tensor
        - batch_size: (int) size of dataset to calculate eNTK in parallel. Larger values take more storage, lower values take longer to calculate

    OUTPUT:
        - eNTK: torch.tensor of size len(x1) x len(x2).
    '''

    if batch_size==0:
        batch_size_1 = len(x1)
        batch_size_2 = len(x2)
    else:
        batch_size_1 = min(batch_size, len(x1))
        batch_size_2 = min(batch_size, len(x2))
    if len(x1) == 1:
        x1_loader = x1
    else:
        x1_loader = DataLoader(x1,batch_size_1)
    if len(x2) == 1:
        x2_loader = x2
    else:
        x2_loader = DataLoader(x2,batch_size_2)

    idx,idy = 0,0
    eNTK = torch.empty((len(x1),len(x2)))
    for i1,x1_batch in enumerate(x1_loader):
        if len(x1_batch)>1:
            x1_batch,_ = x1_batch
        for i2,x2_batch in enumerate(x2_loader):
            if len(x2_batch)>1:
                x2_batch, _ = x2_batch
            
            eNTK_batch = empirical_ntk_jacobian_contraction(fnet_single, params, x1_batch, x2_batch).squeeze((2,3))
            eNTK_shape = eNTK_batch.shape
            eNTK[idx:idx+eNTK_shape[0], idy:idy+eNTK_shape[1]] = eNTK_batch
            idy += eNTK_shape[1]
        idx += eNTK_shape[0]
        idy = 0
    return eNTK


def ntk_uncertainty_explicit_class(train_dataset, test_dataset, model, num_classes, type='direct', rtol=1e-9, maxit=50,softmax=False):
    fnet, params = make_functional(model)    
    uncertainty_array = np.empty((num_classes,len(test_dataset)))
    mu = np.empty((num_classes,len(test_dataset)))
    test_dataloader = DataLoader(test_dataset)
    train_dataloader = DataLoader(train_dataset,len(train_dataset))
    X_train,y_train = next(iter(train_dataloader))

    ### One hot for brier loss
    def one_hot(y,num_classes):
        yh = torch.zeros((y.shape[0],num_classes),dtype=torch.float64)
        yh[torch.arange(y.shape[0]),y] = 1
        return yh

    if softmax:
        f_train = model(X_train).softmax(dim=1)
        y_train = one_hot(y_train,num_classes=num_classes)
    else:
        f_train = model(X_train)
        y_train = one_hot(y_train,num_classes=num_classes)
    resid_hat = (y_train - f_train).detach().numpy()

    with tqdm(total=int(len(test_dataset)*num_classes)) as pbar:
        for c in range(num_classes):
            def fnet_single(params, x):
                if softmax:
                    f = fnet(params, x.unsqueeze(0)).squeeze(0).softmax(dim=0)[c].reshape(1) 
                else:
                    f = fnet(params, x.unsqueeze(0)).squeeze(0)[c].reshape(1) 
                return f
            
            Kappa = empirical_ntk_jacobian_contraction(
                fnet_single=fnet_single,
                params=params,
                x1=X_train,
                x2=X_train
                ).detach().numpy().squeeze((2,3))
    
            for i,(x_test,_) in enumerate(test_dataloader):
                kappa_xx = empirical_ntk_jacobian_contraction(
                    fnet_single=fnet_single,
                    params=params,
                    x1=x_test,
                    x2=x_test
                    ).detach().numpy().squeeze((2,3))
                kappa_xX = empirical_ntk_jacobian_contraction(
                    fnet_single=fnet_single,
                    params=params,
                    x1=x_test,
                    x2=X_train
                    ).detach().numpy().squeeze((2,3))
                if type=='direct':
                    uncertainty_estimate = kappa_xx - kappa_xX @ np.linalg.solve(Kappa,kappa_xX.transpose())
                elif type=='iterative':
                    x_solve, it, resid, rel_resid, rel_mat_resid = solv.CR(
                        Kappa,
                        kappa_xX.transpose(),
                        rtol=rtol,
                        init=False, 
                        maxit=maxit, 
                        VERBOSE=False
                        )
                    
                    # kappa_hat = lifted_solution(x_solve,resid)
                    uncertainty_estimate = kappa_xx - kappa_xX @ x_solve
                    # lifted_ue = kappa_xx - kappa_hat.transpose() @ Kappa @ kappa_hat
                    # uncertainty_array_lift[0,i] = lifted_ue
                    fx = model(x_test)
                    mu_x = x_solve.transpose() @ resid_hat[:,c].reshape((-1,1)) + fx.detach().numpy().squeeze(0)[c]
                uncertainty_array[c,i] = uncertainty_estimate
                mu[c,i] = mu_x
                pbar.update(1)
    
    return uncertainty_array, mu

def ntk_uncertainty_explicit_c(Kappa,train_dataset,test_dataset,model,optimizer,class_c,type='direct',rtol=1e-6,maxit=50):
    uncertainty_array = np.empty((1,len(test_dataset)))
    with tqdm(total=int(len(test_dataset))) as pbar:
        for i,(x,_) in enumerate(test_dataset):
            x = x.reshape((1,28,-1))
            kappa_xx = ntk_matrix_c(x, x, model, optimizer, c=class_c)
            kappa_xX = ntk_matrix_c(x, train_dataset, model, optimizer, c=class_c)
            if type=='direct':
                uncertainty_estimate = kappa_xx - kappa_xX @ np.linalg.solve(Kappa,kappa_xX.transpose())
            elif type=='iterative':
                x_solve,_,_,_,_ = solv.CR(Kappa,kappa_xX.transpose(),rtol=rtol,init=False, maxit=maxit, VERBOSE=False)
                uncertainty_estimate = kappa_xx - kappa_xX @ x_solve
            uncertainty_array[0,i] = uncertainty_estimate
            pbar.update(1)
    return uncertainty_array

def ntk_uncertainty_classification(train_dataset, test_dataset, model, optimizer, num_c,solve_rtol,solve_maxit):
    uncertainty_array = np.empty((num_c,len(test_dataset)))
    with tqdm(num_c) as pbar:
        for c in range(num_c):
            Kappa = ntk_matrix_c(train_dataset,train_dataset,model=model,optimizer=optimizer,c=c)
            uncertainty_array_c = ntk_uncertainty_explicit_c(
                Kappa=Kappa,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                model=model,
                optimizer=optimizer,
                class_c=c,
                type='iterative',
                rtol=solve_rtol,
                maxit=solve_maxit
                )
            uncertainty_array[c,:] = uncertainty_array_c
            pbar.update(1)
    return uncertainty_array

def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2):
        # Compute J(x1)
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        jac1 = [j.flatten(2) for j in jac1]

        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
        jac2 = [j.flatten(2) for j in jac2]
        
        # Compute J(x1) @ J(x2).T
        result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        return result 

def empirical_ntk_jacobian_contraction_c(fnet_single, params, x1, x2,c):
        # Compute J(x1)
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        jac1 = [j[:,c,:].flatten(1) for j in jac1]

        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
        jac2 = [j[:,c,:].flatten(1) for j in jac2]
        
        # Compute J(x1) @ J(x2).T
        result = torch.stack([torch.einsum('Nf,Mf->NM', j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        return result 

# def empirical_ntk(model,dataset1,dataset2):
#     '''
#     Calculates the empirical ntk, which is the Jacobian(model,x1)@Jacobian(model,x2)^T.

#         Parameters:
#             model (PyTorch Model): Neural Network
#             dataset1 (torch.dataset): Object of class dataset, must output torch variables
#             dataset2 (torch.dataset): Object of class dataset, must output torch variables

#         Returns:
#             Kappa (Torch.array): Matrix of size (N1 x N2 x C x C), where N1 = len(dataset1),
#                                 N2 = len(dataset2), C = size(model(xi)).
#     '''
#     fnet, params = make_functional(model)     

#     def fnet_single(params, x):
#         return fnet(params, x.unsqueeze(0)).squeeze(0) 

#     dataset1_ntk = DataLoader(dataset1,len(dataset1))
#     x1_ntk,_ = next(iter(dataset1_ntk))
#     dataset2_ntk = DataLoader(dataset2,len(dataset2))
#     x2_ntk,_ = next(iter(dataset2_ntk))

#     Kappa = empirical_ntk_jacobian_contraction(fnet_single, params, x1_ntk, x2_ntk)
#     return Kappa

def lifted_solution(x,r):
    x = x - ((r.transpose() @ x) / np.linalg.norm(r)) * r
    return x

def lifted_solution_torch(x,r):
    x = x - ((r.transpose(0,1) @ x) / torch.norm(r)) * r
    return x

def tensor_to_block_mat(mat):
    '''
    Mat must have dimension n1 x n2 x n3 x n4, will convert
    to 2d matrix M where at M(i,j) is a n3 x n4 block matrix from tensor.
    '''
    s = mat.shape
    mat = mat.transpose(1,2).reshape(s[0]*s[2],s[1]*s[3])
    return mat