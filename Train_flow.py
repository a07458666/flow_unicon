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
import matplotlib.pyplot as plt

import time
import collections.abc
from collections.abc import MutableMapping
from flow_trainer import FlowTrainer
from flowModule.utils import standard_normal_logprob, linear_rampup
from flowModule.jensen_shannon import js_distance
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
parser.add_argument('--lr_f', '--flow_learning_rate', default=2e-5, type=float, help='initial flow learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha_warmup', default=0.2, type=float, help='parameter for Beta (warmup)')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--linear_u', default=340, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_u', default=3, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_p', default=30, type=float, help='pseudo lamb')
parser.add_argument('--linear_x', default=30, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_x', default=1, type=float, help='weight for supervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--Tu', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--Tx', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=350, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--d_u',  default=0.47, type=float)
parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', action='store_true', help = 'Resume from the warmup checkpoint')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data/cifar10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--flow_modules', default="8-8-8-8", type=str)
parser.add_argument('--name', default="", type=str)
parser.add_argument('--flowRefine', action='store_true')
parser.add_argument('--ce', action='store_true')
parser.add_argument('--fix', default='none', choices=['none', 'net', 'flow'], type=str)
parser.add_argument('--predictPolicy', default='mean', choices=['mean', 'weight'], type=str)
parser.add_argument('--pretrain', default='', type=str)
parser.add_argument('--beta', default=0.1, type=float)
parser.add_argument('--pseudo_std', default=0, type=float)
parser.add_argument('--warmup_mixup', action='store_true')
parser.add_argument('--ema', action='store_true', help = 'Exponential Moving Average')
parser.add_argument('--decay', default=0.995, type=float, help='Exponential Moving Average decay')

args = parser.parse_args()

## GPU Setup
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

## Download the Datasets
if args.dataset== 'cifar10':
    torchvision.datasets.CIFAR10(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR10(args.data_path,train=False, download=True)
elif args.dataset== 'cifar100':
    torchvision.datasets.CIFAR100(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR100(args.data_path,train=False, download=True)

## Checkpoint Location
folder = args.dataset + '_' + args.noise_mode + '_' + str(args.r)  + '_flow_' + args.name
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)
    os.mkdir(model_save_loc + '/JSD_distribution')

## Log files
stats_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     
test_loss_log = open(model_save_loc +'/test_loss.txt','w')
train_acc = open(model_save_loc +'/train_acc.txt','w')
train_loss = open(model_save_loc +'/train_loss.txt','w')
label_count = open(model_save_loc +'/label_count.txt','w')
pu_log = open(model_save_loc +'/pu.txt','w')
loss_log = open(model_save_loc +'/loss_batch.txt','w')

## wandb
if (wandb != None):
    os.environ["WANDB_WATCH"] = "false"
    wandb.init(project="FlowUNICON", entity="andy-su", name=folder)
    wandb.config.update(args)
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("acc", summary="max")

## Test Accuracy
def test(epoch,net,flowNet):
    acc, confidence = flowTrainer.testByFlow(net, flowNet, net_ema, test_loader)
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    
    ## wandb
    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["acc/test"] = acc
        wandb.log(logMsg)
    
    test_log.write(str(acc)+'\n')
    test_log.flush()  
    return acc, confidence

def Selection_Rate(prob):
    threshold = torch.mean(prob)
    if threshold.item()>args.d_u:
        threshold = threshold - (threshold-torch.min(prob))/args.tau
    SR = torch.sum(prob<threshold).item()/num_samples

    if SR <= 1/args.num_class or SR >= 1.0:
        new_SR = np.clip(SR, 1/args.num_class, 0.9)
        print(f'WARNING: sample rate = {SR}, set to {new_SR}')
        SR = new_SR
    return SR, threshold

def logJSD(epoch, threshold, labeled_trainloader, unlabeled_trainloader):
    labeled_idx = labeled_trainloader.dataset.pred_idx
    unlabeled_idx = unlabeled_trainloader.dataset.pred_idx
    origin_prob =  labeled_trainloader.dataset.origin_prob
    labeled_prob = [origin_prob[i] for i in labeled_idx]
    unlabeled_prob = [origin_prob[i] for i in unlabeled_idx]
    sample_ratio = torch.sum(prob<threshold).item()/num_samples

    num_cleanset, num_noiseset = len(labeled_trainloader.dataset), len(unlabeled_trainloader.dataset)
    num_wholeset = num_cleanset + num_noiseset

    cleanset_o_label, cleanset_n_label = labeled_trainloader.dataset.origin_label, labeled_trainloader.dataset.noise_label
    noiseset_o_label, noiseset_n_label = unlabeled_trainloader.dataset.origin_label, unlabeled_trainloader.dataset.noise_label

    cleanset_noise_mask = (cleanset_o_label != cleanset_n_label).astype(float)
    noiseset_noise_mask = (noiseset_o_label != noiseset_n_label).astype(float)
    
    num_cleanset_noise = cleanset_noise_mask.sum()
    num_noiseset_noise = noiseset_noise_mask.sum()
    num_noise = num_cleanset_noise + num_noiseset_noise

    num_cleanset_clean = num_cleanset - num_cleanset_noise
    num_noiseset_clean = num_noiseset - num_noiseset_noise
    num_clean = num_wholeset - num_noise

    eps = 1e-20
    clean_recall = num_cleanset_clean / (num_clean + eps)
    clean_precision = num_cleanset_clean / (num_cleanset + eps)
    clean_f1 = (2 * clean_recall * clean_precision) / (clean_recall + clean_precision + eps)

    noise_recall = num_noiseset_noise / (num_noise + eps)
    noise_precision = num_noiseset_noise / (num_noiseset + eps)
    noise_f1 = (2 * noise_recall * noise_precision) / (noise_recall + noise_precision + eps)

    # draw JSD dis
    clean_prob = []
    noise_prob = []
    for idx_noise_zip in [zip(labeled_idx, cleanset_noise_mask), zip(unlabeled_idx, noiseset_noise_mask)]:
        for idx, is_noise in idx_noise_zip:
            p = origin_prob[idx]
            if is_noise == 1.0:
                noise_prob.append(float(p))
            else:
                clean_prob.append(float(p))

    plt.clf()
    kwargs = dict(histtype='stepfilled', alpha=0.75, density=False, bins=20)
    plt.hist(clean_prob, color='green', range=(0., 1.), label='clean', **kwargs)
    plt.hist(noise_prob, color='red'  , range=(0., 1.), label='noisy', **kwargs)

    plt.axvline(x=threshold,          color='black')
    plt.axvline(x=origin_prob.mean(), color='gray')
    plt.xlabel('JSD Values')
    plt.ylabel('count')
    plt.title(f'JSD Distribution of N Samples epoch :{epoch}')
    plt.xlim(0, 1)
    plt.grid(True)
    plt.savefig(f'{model_save_loc}/JSD_distribution/epoch{epoch}.png')

    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["JSD"] = wandb.Image(f'{model_save_loc}/JSD_distribution/epoch{epoch}.png')
        logMsg["JSD/threshold"] = threshold
        logMsg["JSD/sample_ratio"] = sample_ratio
        logMsg["JSD/labeled_mean"] =  np.mean(labeled_prob)
        logMsg["JSD/labeled_var"] = np.var(labeled_prob)
        logMsg["JSD/unlabeled_mean"] = np.mean(unlabeled_prob)
        logMsg["JSD/unlabeled_var"] = np.var(unlabeled_prob)

        logMsg["JSD_clean/labeled_mean"] =  np.mean(labeled_prob)
        logMsg["JSD_clean/labeled_var"] = np.var(labeled_prob)
        logMsg["JSD_clean/unlabeled_mean"] = np.mean(unlabeled_prob)
        logMsg["JSD_clean/unlabeled_var"] = np.var(unlabeled_prob)

        logMsg["JSD_selection/clean_recall"] = clean_recall
        logMsg["JSD_selection/clean_precision"] = clean_precision
        logMsg["JSD_selection/clean_f1"] = clean_f1
    
        logMsg["JSD_selection/noise_recall"] = noise_recall
        logMsg["JSD_selection/noise_precision"] = noise_precision
        logMsg["JSD_selection/noise_f1"] = noise_f1
        wandb.log(logMsg)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch,warm_up, args.lambda_u)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

def logFeature(feature):
    ## wandb
    if (wandb != None):
        logMsg = {}
        logMsg["Feature/max"] = feature.max()
        logMsg["Feature/min"] = feature.min()
        logMsg["Feature/mean"] = feature.mean()
        logMsg["Feature/var"] = feature.var()
        logMsg["Feature/var_dim"] = feature.var(dim=0).mean()
        wandb.log(logMsg)
    return

def log_pu(pu_flow, pu_net, gt):
    prob_flow, predicted_flow = torch.max(pu_flow, 1)
    prob_net, predicted_net = torch.max(pu_net, 1)

    total = gt.size(0)
    correct_flow = predicted_flow.eq(gt).cpu().sum().item()  
    prob_sum_flow = prob_flow.cpu().sum().item()

    correct_net = predicted_net.eq(gt).cpu().sum().item()  
    prob_sum_net = prob_net.cpu().sum().item()

    acc_flow = 100.*correct_flow/total
    confidence_flow = prob_sum_flow/total

    acc_net = 100.*correct_net/total
    confidence_net = prob_sum_net/total

    pu_log.write('\nepoch : ' + str(epoch))
    pu_log.write('\n acc_flow : ' + str(acc_flow))
    pu_log.write('\n confidence_flow : ' + str(confidence_flow))
    
    pu_log.write('\n acc_net : ' + str(acc_net))
    pu_log.write('\n confidence_net : ' + str(confidence_net))

    pu_log.write('\n pu_flow : ' + str(pu_flow[:5].cpu().numpy()))
    pu_log.write('\n pu_net : ' + str(pu_net[:5].cpu().numpy()))
    pu_log.write('\n gt : ' + str(gt[:5].cpu().numpy()))
    
    pu_log.flush()

    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["pseudo/acc_flow"] = acc_flow
        logMsg["pseudo/confidence_flow"] = confidence_flow
        logMsg["pseudo/acc_net"] = acc_net
        logMsg["pseudo/confidence_net"] = confidence_net
        wandb.log(logMsg)
    return

def print_label_status(targets_x, targets_u, labels_x_o, labels_u_o, batch_idx):
    label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    refine_labels_x = [0] * args.num_class
    target_labels_x = [0] * args.num_class

    pseudo_labels_u = [0] * args.num_class
    target_labels_u = [0] * args.num_class
    for i in targets_u.max(dim=1).indices:
        pseudo_labels_u[i.item()] += 1
    for i in labels_u_o:
        target_labels_u[i.item()] += 1

    for i in targets_x.max(dim=1).indices:
        refine_labels_x[i.item()] += 1
    for i in labels_x_o:
        target_labels_x[i.item()] += 1
    label_count.write('\nepoch : ' + str(epoch))
    label_count.write('\npseudo_labels_u : ' + str(pseudo_labels_u))
    label_count.write('\ntarget_labels_u : ' + str(target_labels_u))
    label_count.write('\nrefine_labels_x : ' + str(refine_labels_x))
    label_count.write('\ntarget_labels_x : ' + str(target_labels_x))
    label_count.flush()

    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["label_count/pseudo_labels_u"] =  max(pseudo_labels_u)
        logMsg["label_count/target_labels_u"] =  max(target_labels_u)
        logMsg["label_count/refine_labels_x"] =  max(refine_labels_x)
        logMsg["label_count/target_labels_x"] =  max(target_labels_x)
        wandb.log(logMsg)

def save_model(net, flowNet, epoch, model_name, model_name_flow, acc = 0):
    print("Save the Model-----")
    checkpoint = {
        'net': net.state_dict(),
        'Model_number': 1,
        'Noise_Ratio': args.r,
        'Loss Function': 'CrossEntropyLoss',
        'Optimizer': 'SGD',
        'Noise_mode': args.noise_mode,
        'Accuracy': acc,
        'Pytorch version': '1.4.0',
        'Dataset': 'TinyImageNet',
        'Batch Size': args.batch_size,
        'epoch': epoch,
    }

    checkpoint_flow = {
        'net': flowNet.state_dict(),
        'Model_number': 3,
        'Noise_Ratio': args.r,
        'Loss Function': 'log-likelihood',
        'Optimizer': 'SGD',
        'Noise_mode': args.noise_mode,
        'Dataset': 'TinyImageNet',
        'Batch Size': args.batch_size,
        'epoch': epoch,
    }

    torch.save(checkpoint, os.path.join(model_save_loc, model_name))
    torch.save(checkpoint_flow, os.path.join(model_save_loc, model_name_flow))

## Choose Warmup period based on Dataset
num_samples = 50000
if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30

## Call the dataloader
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=4,\
    root_dir=model_save_loc,log=stats_log, noise_file='%s/clean_%.4f_%s.npz'%(args.data_path,args.r, args.noise_mode))

print('| Building net')
net = create_model()

# flow model
flowTrainer = FlowTrainer(args, warm_up)
flowNet = flowTrainer.create_model()

cudnn.benchmark = True

## Semi-Supervised Loss
criterion  = SemiLoss()

## Optimizer and Scheduler
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
optimizerFlow = optim.SGD(flowNet.parameters(), lr=args.lr_f, momentum=0.9, weight_decay=5e-4)
# optimizerFlow = optim.AdamW(flowNet.parameters(), lr=args.lr_f)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, args.lr / 100)
schedulerFlow = optim.lr_scheduler.CosineAnnealingLR(optimizerFlow, args.num_epochs, args.lr_f / 100)

if args.ema:
    net_ema = ExponentialMovingAverage(net.parameters(), decay=args.decay)
    flowNet_ema = ExponentialMovingAverage(flowNet.parameters(), decay=args.decay)
    flowTrainer.setEma(flowNet_ema)
else:
    net_ema = None
    flowNet_ema = None

## Resume from the warmup checkpoint 
model_name = 'Net_warmup.pth'
model_name_flow = 'FlowNet_warmup.pth'

if args.resume:
    start_epoch = warm_up
    net.load_state_dict(torch.load(os.path.join(model_save_loc, model_name))['net'])
    flowNet.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_flow))['net'])
elif args.pretrain != '':
    start_epoch = 0
    warm_up = 1
    net.load_state_dict(torch.load(args.pretrain)['net'])
else:
    start_epoch = 0

best_acc = 0

## Warmup and SSL-Training 
for epoch in range(start_epoch,args.num_epochs+1):
    startTime = time.time() 
    test_loader = loader.run(0, 'test')
    eval_loader = loader.run(0, 'eval_train')   
    warmup_trainloader = loader.run(0,'warmup')
    
    ## Warmup Stage 
    if epoch<warm_up:       
        warmup_trainloader = loader.run(0, 'warmup')

        print('Warmup Model')
        warmup_standard(epoch, net, flowNet, net_ema, flowNet_ema, optimizer, optimizerFlow, warmup_trainloader)   
    
    else:
        ## Calculate JSD values and Filter Rate
        print("Calculate JSD")
        prob = Calculate_JSD(net, flowNet, num_samples)
        SR , threshold = Selection_Rate(prob)
        
        print('Train Net\n')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= prob) # Uniform Selection
        logJSD(epoch, threshold, labeled_trainloader, unlabeled_trainloader)
        flowTrainer.train(epoch, net, flowNet, net_ema, flowNet_ema, optimizer, optimizerFlow, labeled_trainloader, unlabeled_trainloader)    # train net1  

    acc, confidence = test(epoch,net, flowNet)

    scheduler.step()
    schedulerFlow.step()

    ## wandb
    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["runtime"] = time.time() - startTime
        logMsg["confidence score"] = confidence
        wandb.log(logMsg)

    if acc > best_acc:
        if epoch <warm_up:
            model_name = 'Net_warmup.pth'
            model_name_flow = 'FlowNet_warmup.pth'
        else:
            model_name = 'Net.pth'
            model_name_flow = 'FlowNet.pth'

        save_model(net, flowNet, epoch, model_name, model_name_flow, acc)
        best_acc = acc

