'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import pickle

import torchvision
import torchvision.transforms as transforms
#from imagenet32 import ImageNet32
#from cifar_random import CIFAR10RandomLabels

import os
import argparse
from random import randrange

from models import *

parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--models', nargs='+',type=str, help='models', required=True)
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--dataset',default='cifar',type=str,help='dataset (cifar or imagenet)')
parser.add_argument('--bs', default=64, type=int, help='batch size')
parser.add_argument('--num_epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--subsample', default=None, type=int, help='number in subsampled dataset')
parser.add_argument('--noisedata',action="store_true",help='replace data with noise')
parser.add_argument('--repeats',default=1, type=int, help='number of times to train the model')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_epochs = args.num_epochs
norms = []

def img_colorize(x):
    return x.repeat(3, 1, 1)

# Data
print('==> Preparing data..')

if args.dataset == 'cifar':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.noisedata:
        print('Preparing noise data...')
        trainset = CIFAR10RandomLabels(
            root='./cifar_data', train=True, download=True, transform=transform_train,corrupt_prob=0.9)
    else:
        trainset = torchvision.datasets.CIFAR10(
            root='./cifar_data', train=True, download=True, transform=transform_train)
    if args.subsample is not None:
        trainset = torch.utils.data.Subset(trainset, list(range(args.subsample)))
        with open(args.dataset+'_trainset.pkl','wb') as handle:
            pickle.dump(trainset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True)
    trainloader_big = torch.utils.data.DataLoader(
        trainset, batch_size=1024, shuffle=True)
    testset = torchvision.datasets.CIFAR10(
        root='./cifar_data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = 10
elif args.dataset == 'mnist':
    transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.Resize((32,32)),
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Lambda(img_colorize)])
    trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
    if args.subsample is not None:
        trainset = torch.utils.data.Subset(trainset, list(range(args.subsample)))
        with open(args.dataset+'_trainset.pkl','wb') as handle:
            pickle.dump(trainset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True)
    testset = torchvision.datasets.MNIST(
        root='./mnist_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False)
    num_classes = 10
elif args.dataset == 'fmnist':
    transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.Resize((32,32)),
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Lambda(img_colorize)])
    trainset = torchvision.datasets.FashionMNIST(root='./fmnist_data', train=True, download=True, transform=transform)
    if args.subsample is not None:
        trainset = torch.utils.data.Subset(trainset, list(range(args.subsample)))
        with open(args.dataset+'_trainset.pkl','wb') as handle:
            pickle.dump(trainset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True)
    testset = torchvision.datasets.FashionMNIST(
        root='./fmnist_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False)
    num_classes = 10
elif args.dataset == 'svhn':
    transform = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    trainset = torchvision.datasets.SVHN(root='./svhn_data', split='train', download=True, transform=transform)
    if args.subsample is not None:
        trainset = torch.utils.data.Subset(trainset, list(range(args.subsample)))
        with open(args.dataset+'_trainset.pkl','wb') as handle:
            pickle.dump(trainset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True)
    testset = torchvision.datasets.SVHN(
        root='./svhn_data', split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False)
    num_classes = 10
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4811,0.4575,0.4079),
        #                     (0.2604,0.2532,0.2682)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4811,0.4575,0.4079),
        #                     (0.2604,0.2532,0.2682)),
    ])
    target_transform = lambda y: torch.randint(0, 10, (1,)).item()
    trainloader = None
    trainset = ImageNet32('./Imagenet32_train/',train=True,transform=transform_train)
    if args.subsample is not None:
        trainset = torch.utils.data.Subset(trainset, list(range(args.subsample)))
        with open(args.dataset+'_trainset.pkl','wb') as handle:
            pickle.dump(trainset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    testset = ImageNet32('./Imagenet32_val/',train=False,transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False)
    num_classes = 1000
    num_epochs *= 10 # Due to swapping between 10 banks of images

#names = ['wrn-28-5']
#names = ['resnet18']
#names = ['resnet9','resnet18','resnet34','resnet68']

for name in args.models:
    for rep in range(args.repeats):
        pathname = '{}_{}_{}'.format(args.dataset,name,args.subsample)
        chkpath = 'ckpt/{}.pth'.format(pathname)
        #chkpointdir = args.dataset + '_' + name
        #chkpointpath = './'+chkpointdir+'/ckpt_'+str(rep)+'.pth'
        start_epoch = 0
        print('====================')
        print('==> Building model', name)
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
        elif name == 'wrn-28-2':
            net = WRN(depth=28, widening_factor=2, num_classes = num_classes)
        elif name == 'wrn-28-5':
            net = WRN(depth=28, widening_factor=5, num_classes = num_classes)
        elif name == 'wrn-28-10':
            net = WRN(depth=28, widening_factor=10, num_classes = num_classes)
        elif name == 'logistic':
            net = Logistic()
        elif name == 'twolayer':
            net = TwoLayer()
        elif name == 'twolayermini':
            net = TwoLayerBottle()
        elif name == 'densenet121':
            net = DenseNet121()
        net = net.to(device)
        print('number of parameters:', sum(p.numel() for p in net.parameters()))
        if device == 'cuda':
            cudnn.benchmark = True
        # if os.path.isdir(chkpointdir):
        #     checkpoint = torch.load(chkpointpath)
        #     start_epoch = checkpoint['epoch']
        #     if start_epoch+1 >= args.num_epochs:
        #         print('==> Completed')
        #         continue
        #     net.load_state_dict(checkpoint['net'])

        criterion = nn.CrossEntropyLoss()
        #criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        for idx in range(start_epoch):
            scheduler.step()

        # Training
        def train(epoch):
            if args.dataset == 'imagenet':
                print('\nEpoch: %d.%d' % (epoch // 10, epoch % 10))
            else:
                print('\nEpoch: %d' % epoch)
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            epoch_norms = []
            if args.dataset == 'imagenet':
                # trainset.load(randrange(10)) # Switch ImageNet bank at random
                trainset.load(epoch % 10) # Switch ImageNet bank orderly
            # if epoch > 0.5*num_epochs:
            #     trainloader = torch.utils.data.DataLoader(
            #         trainset, batch_size=1024, shuffle=True)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=args.bs, shuffle=True)
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                total_norm = 0
                for p in net.parameters():
                    param_norm = torch.sum(p.grad.data**2).item()
                    if p.var() < 1e-9:
                        continue
                    total_norm += param_norm
                total_norm = total_norm ** (1. / 2)
                #print(total_norm)
                epoch_norms.append(total_norm)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if (batch_idx % 10) == 0:
                    print(name,'|',batch_idx,'of', len(trainloader), '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            #norms.append(epoch_norms)
            #np.save(name+'_norms.npy',norms)
            return correct/total


        def test(epoch):
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                #if epoch >= 0.5*num_epochs:
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    if (batch_idx % 10) == 0:
                        print(name,'|',batch_idx, 'of', len(testloader), '| Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # Save checkpoint.
            if total > 0:
                acc = 100.*correct/total
            else:
                acc = 0
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'loss': test_loss,
                'acc': acc,
                'epoch': epoch,
            }
            # if not os.path.isdir(chkpointdir):
            #     os.mkdir(chkpointdir)
            torch.save(state, chkpath)


        for epoch in range(start_epoch, num_epochs):
            acc = train(epoch)
            #if acc > 0.98:
            if acc == 1 or epoch > num_epochs - 5:
                test(epoch)
                if acc == 1:
                    break
            scheduler.step()
