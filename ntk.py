import torch
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
from functorch import make_functional_with_buffers, vmap, jacrev
#from torch import vmap#, jacrev
#from torch.func import jacrev
#from torch.func import functional_call as make_functional_with_buffers
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
from models import *
import numpy as np
import os
import argparse

import pickle

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--models', nargs='+',type=str, help='models', required=True)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset',default='cifar',type=str,help='dataset (cifar or imagenet)')
parser.add_argument('--bs', default=16, type=int, help='batch size')
parser.add_argument('--num_epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--subsample', default=None, type=int, help='number in subsampled dataset')
parser.add_argument('--noisedata',action="store_true",help='replace data with noise')
parser.add_argument('--repeats',default=1, type=int, help='number of times to train the model')
parser.add_argument('--init',action="store_true",help='use initialized model')
args = parser.parse_args()

if torch.cuda.is_available() and torch.version.hip:
    print('HIP is working!')

if torch.cuda.is_available():
    device = 'cuda'
    print('CUDA is on')
# elif torch.backends.mps.is_available():
#     device = 'mps'
#     print('MPS is on')
else:
    print('CPU is used')
    device = 'cpu'

num_classes = 10

# def make_functional_with_buffers(mod, disable_autograd_tracking=False):
#     params_dict = dict(mod.named_parameters())
#     params_names = params_dict.keys()
#     params_values = tuple(params_dict.values())

#     buffers_dict = dict(mod.named_buffers())
#     buffers_names = buffers_dict.keys()
#     buffers_values = tuple(buffers_dict.values())
    
#     stateless_mod = copy.deepcopy(mod)
#     stateless_mod.to('meta')

#     def fmodel(new_params_values, new_buffers_values, *args, **kwargs):
#         new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
#         new_buffers_dict = {name: value for name, value in zip(buffers_names, new_buffers_values)}
#         return torch.func.functional_call(stateless_mod, (new_params_dict, new_buffers_dict), args, kwargs)
  
#     if disable_autograd_tracking:
#         params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
#     return fmodel, params_values, buffers_values

class NTK(object):
    def __init__(self, net, num_classes, name):
        self.name = name
        self.net = net.to(device)
        self.net.eval()
        self.fnet, self.params, self.buffers = make_functional_with_buffers(net)
        self.num_classes = num_classes
        
    def fnet_single(self, params, buffers, x):
        return self.fnet(params, buffers, x.unsqueeze(0)).squeeze(0)
    
    def empirical_ntk(self, fnet_single, params, x1, x2):
        # Compute J(x1)
        jac1 = vmap(jacrev(self.fnet_single), (None, None, 0))(params, self.buffers, x1)
        jac1 = [j.detach().flatten(2).flatten(0,1) for j in jac1]
        #jac1 = [j.detach().flatten(1) for j in jac1]

        # Compute J(x2)
        jac2 = vmap(jacrev(self.fnet_single), (None, None, 0))(params, self.buffers, x2)
        jac2 = [j.detach().flatten(2).flatten(0,1) for j in jac2]
        #jac2 = [j.detach().flatten(1) for j in jac2]

        # Compute J(x1) @ J(x2).T
        result = torch.stack([torch.einsum('Nf,Mf->NM', j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        return result
    
    def __call__(self, trainloader):
        # Get number of batches
        num_batches = len(trainloader)
        #start = np.memmap('ntk_idx.npy',dtype='int32',mode='w+',shape=(2,))
        dataset = [(idx, x) for idx, (x, _) in enumerate(trainloader)]
        # Get number of data points
        N = 0
        N = sum([x.shape[0] for (idx, x) in dataset])
        ntk = np.memmap(self.name+'_ntk.npy', dtype='float64', mode='w+', shape=(N*self.num_classes,N*self.num_classes))
        nidx = 0
        nidy = 0
        with tqdm(total=int(len(dataset)*(len(dataset)+1)/2)) as pbar:
            for idx, x1 in dataset:
                for idy, x2 in dataset:
                    if idy < idx:
                        nidy += x2.shape[0]*self.num_classes
                        continue
                    J = self.empirical_ntk(self.fnet_single, self.params, x1.to(device), x2.to(device)).cpu()
                    incx = J.shape[0]
                    incy = J.shape[1]
                    ntk[nidx:nidx+incx,nidy:nidy+incy] = J
                    if idx != idy:
                        ntk[nidy:nidy+incy,nidx:nidx+incx] = J.T
                    nidy += incy
                    pbar.update(1)
                nidx += incx
                nidy = 0
        return ntk
    
class CIFARTrainer(object):
    
    def __init__(self, net, lr=1e-2, subsample=None):
        self.net = net
        #self.criterion = nn.MSELoss(reduction='mean')
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        self.dim_d = self.count_parameters()
        self.subsample = subsample
        if subsample is None:
            self.dim_n = 50000 * 10
        else:
            self.dim_n = subsample * 10
        if self.dim_d > self.dim_n:
            print('Overparameterized')
        else:
            print('Underparameterized')
        print('d =', self.dim_d)
        print('n =', self.dim_n)
        
    def prepare_dataset(self, bs=64, corrupt_prob=None):
        with open(args.dataset+'_trainset.pkl','rb') as handle:
            self.trainset = pickle.load(handle)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=bs, shuffle=True)
        if args.dataset == 'cifar' or args.dataset == 'svhn':
            self.num_classes = 10
        else:
            self.num_classes = 1000

    def train(self,epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.double().to(device), targets.double().to(device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            #loss = self.criterion(outputs, F.one_hot(targets, num_classes = self.num_classes).double())
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx % 10) == 0:
                print('CNN','|',batch_idx,'of', len(self.trainloader), '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return correct/total
    
    def full_train(self):
        for epoch in range(0, 200):
            acc = self.train(epoch)
            self.scheduler.step()
            if acc == 1:
                break
            clear_output(wait=True)
            
    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
    
for name in args.models:
    for rep in range(args.repeats):
        pathname = '{}_{}_{}'.format(args.dataset,name,args.subsample)
        chkpath = 'ckpt/{}.pth'.format(pathname)
        ntk_name = args.dataset + '_' + name + '_' + str(rep) + '_' + str(args.subsample)
        if args.init:
            ntk_name += '_init'
        if os.path.isfile(ntk_name + '_eigs.npy'):
            continue
        #chkpointpath = './'+chkpointdir+'/ckpt_'+str(rep)+'.pth'
        if name == 'resnet9':
            net = ResNet9(num_classes=num_classes)
        elif name == 'resnet18':
            net = ResNet18(num_classes=num_classes)
        elif name == 'resnet34':
            net = ResNet34(num_classes=num_classes)
        elif name == 'resnet50':
            net = ResNet50(num_classes=num_classes)
        elif name == 'resnet68':
            net = ResNet68(num_classes=num_classes)
        elif name == 'resnet101':
            net = ResNet101(num_classes=num_classes)
        elif name == 'resnet152':
            net = ResNet152(num_classes=num_classes)
        elif name == 'mobilenet':
            net = MobileNet(num_classes=num_classes)
        elif name == 'mobilenetv2':
            net = MobileNetV2(num_classes=num_classes)
        elif name == 'vgg11':
            net = VGG('VGG11')
        elif name == 'vgg13':
            net = VGG('VGG13')
        elif name == 'lenet':
            net = LeNet()
        elif name == 'logistic':
            net = Logistic()
        elif name == 'twolayer':
            net = TwoLayer()
        elif name == 'twolayermini':
            net = TwoLayerBottle()
        elif name == 'densenet121':
            net = DenseNet121()
        if os.path.isfile(chkpath) and not args.init:
            checkpoint = torch.load(chkpath, map_location=device)
            net.load_state_dict(checkpoint['net'])
            print('Model loaded')
        net = net.to(device).double()
        trainer = CIFARTrainer(net, lr=args.lr, subsample=args.subsample)
        K = NTK(net, num_classes, ntk_name)
        trainer.prepare_dataset(args.bs)
        ntk = K(trainer.trainloader)
        eigs = np.linalg.eigvalsh(ntk)
        np.save(ntk_name + '_eigs.npy', eigs)
        os.remove(ntk_name + '_ntk.npy')
        
