from __future__ import print_function
import imp
import sys
from syslog import LOG_MAIL

from urllib3 import Retry
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import random
import os
import argparse
import numpy as np
from PreResNet_cifar import *
import dataloader_cifar as dataloader
from math import log2
from Contrastive_loss import *
import matplotlib.pyplot as plt

import collections.abc
from collections.abc import MutableMapping
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

## For plotting the logs
# import wandb
# wandb.init(project="noisy-label-project", entity="..")

## Arguments to pass 
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--num_epochs', default=350, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data/cifar10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--name', default="", type=str)
parser.add_argument('--wandb', action='store_false')
args = parser.parse_args()

if not args.wandb:
    wandb = None

## GPU Setup 
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

## Download the Datasets
if args.dataset== 'cifar10':
    torchvision.datasets.CIFAR10(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR10(args.data_path,train=False, download=True)
else:
    torchvision.datasets.CIFAR100(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR100(args.data_path,train=False, download=True)

## Checkpoint Location
folder = args.dataset + '_' + args.noise_mode + '_ssl_' + args.name
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)
    os.mkdir(model_save_loc + '/class_distribution')

## Log files
stats_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     
test_loss_log = open(model_save_loc +'/test_loss.txt','w')
train_acc = open(model_save_loc +'/train_acc.txt','w')
train_loss = open(model_save_loc +'/train_loss.txt','w')

## wandb
if (wandb != None):
    os.environ["WANDB_WATCH"] = "false"
    wandb.init(project="FlowUNICON", entity="andy-su", name=folder)
    wandb.config.update(args)
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("acc", summary="max")

# SSL-Training
def train(epoch, net, optimizer, trainloader):
    net.train()

    num_iter = (len(trainloader.dataset)//args.batch_size)+1

    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x) in enumerate(trainloader):      

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda()

        ## Unsupervised Contrastive Loss
        f1, _ = net(inputs_x3)
        f2, _ = net(inputs_x4)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss_simCLR = contrastive_criterion(features)
                        
        ## Total Loss
        loss = loss_simCLR

        # Compute gradient and Do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## wandb
        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["loss/simCLR"] = loss_simCLR.item()
            wandb.log(logMsg)

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Contrastive Loss:%.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss_simCLR.item()))
        sys.stdout.flush()

# def logDistubtion(epoch, trainloader):
#     plt.clf()
#     kwargs = dict(histtype='stepfilled', alpha=0.75, density=False, bins=20)
#     plt.hist(clean_prob, color='green', range=(0., 1.), label='clean', **kwargs)
#     plt.hist(noise_prob, color='red'  , range=(0., 1.), label='noisy', **kwargs)

#     plt.axvline(x=threshold,          color='black')
#     plt.axvline(x=origin_prob.mean(), color='gray')
#     plt.xlabel('JSD Values')
#     plt.ylabel('count')
#     plt.title(f'JSD Distribution of N Samples epoch :{epoch}')
#     plt.xlim(0, 1)
#     plt.grid(True)
#     plt.savefig(f'{model_save_loc}/class_distribution/epoch{epoch}.png')
#     return

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

## Choose Warmup period based on Dataset
num_samples = 50000

## Call the dataloader
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=4,\
    root_dir=model_save_loc,log=stats_log, noise_file='%s/clean_%.4f_%s.npz'%(args.data_path,args.r, args.noise_mode))

print('| Building net')
net = create_model()

cudnn.benchmark = True


## Optimizer and Scheduler
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, args.lr / 100)

## Loss Functions
contrastive_criterion = SupConLoss()

model_name = 'Net_ssl.pth'

start_epoch = 0

best_loss = 0

for epoch in range(start_epoch,args.num_epochs+1):   
    test_loader = loader.run(0, 'test')
    eval_loader = loader.run(0, 'eval_train')   
    _, trainloader = loader.run(1.0, 'train')
    
    print('Train Net\n')
    loss = train(epoch, net, optimizer, trainloader)

    scheduler.step()

    if loss > best_loss:
        model_name = 'Net.pth'

        print("Save the Model-----")
        checkpoint = {
            'net': net.state_dict(),
            'Model_number': 1,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Loss': loss,
            'Pytorch version': '1.4.0',
            'Dataset': 'TinyImageNet',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        torch.save(checkpoint, os.path.join(model_save_loc, model_name))
        best_loss = loss

