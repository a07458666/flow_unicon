from __future__ import print_function

import os
from syslog import LOG_MAIL
from urllib3 import Retry
import random
import numpy as np
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models

# 
import time
import collections.abc
from flow_trainer import FlowTrainer
from tqdm import tqdm
from config import argumentParse

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

## For plotting the logs
# import wandb
# wandb.init(project="noisy-label-project", entity="..")

## Arguments to pass 
args = argumentParse()
print("args : ",vars(args))

## GPU Setup
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

## Download the Datasets
if args.dataset== 'cifar10' or args.dataset== 'cifar100':
    from dataloader_cifar import cifar_dataloader as dataloader
    from PreResNet_cifar import *
    if args.dataset == 'cifar10':
        torchvision.datasets.CIFAR10(args.data_path,train=True, download=True)
        torchvision.datasets.CIFAR10(args.data_path,train=False, download=True)
    else:
        torchvision.datasets.CIFAR100(args.data_path,train=True, download=True)
        torchvision.datasets.CIFAR100(args.data_path,train=False, download=True)
elif args.dataset=='TinyImageNet':
    from PreResNet_tiny import *
    from dataloader_tiny import tinyImagenet_dataloader as dataloader
elif args.dataset=="Clothing1M":
    from PreResNet_clothing1M import *
    import dataloader_clothing1M as dataloader

## Checkpoint Location
folder = args.dataset + '_' + args.noise_mode + '_' + str(args.ratio)  + '_flow_' + args.name
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)
    os.mkdir(model_save_loc + '/JSD_distribution')

## Log files
stats_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.ratio,args.noise_mode)+'_stats.txt','w') 
test_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.ratio,args.noise_mode)+'_acc.txt','w')     
test_loss_log = open(model_save_loc +'/test_loss.txt','w')
train_acc = open(model_save_loc +'/train_acc.txt','w')
train_loss = open(model_save_loc +'/train_loss.txt','w')
label_count = open(model_save_loc +'/label_count.txt','w')
pu_log = open(model_save_loc +'/pu.txt','w')
loss_log = open(model_save_loc +'/loss_batch.txt','w')

## wandb
if (wandb != None):
    wandb.init(project="FlowUNICON", entity="andy-su", name=folder)
    wandb.run.log_code(".")
    wandb.config.update(args)
    wandb.define_metric("acc/test", summary="max")
    wandb.define_metric("loss/nll", summary="min")
    wandb.define_metric("loss/nll_max", summary="min")
    wandb.define_metric("loss/nll_min", summary="min")
    wandb.define_metric("loss/nll_var", summary="min")

def Selection_Rate(prob, pre_threshold):
    threshold = torch.mean(prob)
    SR = torch.sum(prob<threshold).item()/args.num_samples
    if args.ema_jsd:
        threshold = (args.jsd_decay * pre_threshold) + ((1 - args.jsd_decay) * threshold)
    print("threshold : ", torch.mean(prob))
    print("threshold(new) : ", threshold)
    print("prob size : ", prob.size())
    print("down :", torch.sum(prob<threshold).item())
    print("up :", torch.sum(prob>threshold).item())
    if SR <= 0.1  or SR >= 1.0:
        new_SR = np.clip(SR, 0.1 , 0.9)
        print(f'WARNING: sample rate = {SR}, set to {new_SR}')
        SR = new_SR
    return SR, threshold

def logJSD(epoch, threshold, labeled_trainloader, unlabeled_trainloader):
    labeled_idx = labeled_trainloader.dataset.pred_idx
    unlabeled_idx = unlabeled_trainloader.dataset.pred_idx
    origin_prob =  labeled_trainloader.dataset.origin_prob
    labeled_prob = [origin_prob[i] for i in labeled_idx]
    unlabeled_prob = [origin_prob[i] for i in unlabeled_idx]
    sample_ratio = torch.sum(prob<threshold).item()/args.num_samples

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

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

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
    if epoch <args.warm_up:
        model_name = 'Net_warmup.pth'
        model_name_flow = 'FlowNet_warmup.pth'
        model_name_ema = 'Net_warmup_ema.pth'
        model_name_flow_ema = 'FlowNet_warmup_ema.pth'
    else:
        model_name = 'Net.pth'
        model_name_flow = 'FlowNet.pth'
        model_name_ema = 'Net_ema.pth'
        model_name_flow_ema = 'FlowNet_ema.pth'

    print("Save the Model-----")
    checkpoint = {
        'net': net.state_dict(),
        'Model_number': 1,
        'Noise_Ratio': args.ratio,
        'Loss Function': 'CrossEntropyLoss',
        'Optimizer': 'SGD',
        'Noise_mode': args.noise_mode,
        'Accuracy': acc,
        'Pytorch version': '1.4.0',
        'Dataset': args.dataset,
        'Batch Size': args.batch_size,
        'epoch': epoch,
    }

    checkpoint_flow = {
        'net': flowNet.state_dict(),
        'Model_number': 2,
        'Noise_Ratio': args.ratio,
        'Loss Function': 'log-likelihood',
        'Optimizer': 'SGD',
        'Noise_mode': args.noise_mode,
        'Dataset': args.dataset,
        'Batch Size': args.batch_size,
        'epoch': epoch,
    }
    
    checkpoint_ema = {
        'net': flowTrainer.net_ema.state_dict(),
        'Model_number': 3,
        'Noise_Ratio': args.ratio,
        'Loss Function': 'CrossEntropyLoss',
        'Optimizer': 'SGD',
        'Noise_mode': args.noise_mode,
        'Accuracy': acc,
        'Pytorch version': '1.4.0',
        'Dataset': args.dataset,
        'Batch Size': args.batch_size,
        'epoch': epoch,
    }

    checkpoint_flow_ema = {
        'net': flowTrainer.flowNet_ema.state_dict(),
        'Model_number': 4,
        'Noise_Ratio': args.ratio,
        'Loss Function': 'log-likelihood',
        'Optimizer': 'SGD',
        'Noise_mode': args.noise_mode,
        'Dataset': args.dataset,
        'Batch Size': args.batch_size,
        'epoch': epoch,
    }

    torch.save(checkpoint, os.path.join(model_save_loc, model_name))
    torch.save(checkpoint_flow, os.path.join(model_save_loc, model_name_flow))
    
    torch.save(checkpoint_ema, os.path.join(model_save_loc, model_name_ema))
    torch.save(checkpoint_flow_ema, os.path.join(model_save_loc, model_name_flow_ema))

## Call the dataloader
if args.dataset== 'cifar10' or args.dataset== 'cifar100':
    loader = dataloader(args.dataset, r=args.ratio, noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=4,\
        root_dir=model_save_loc, noise_file='%s/clean_%.4f_%s.npz'%(args.data_path,args.ratio, args.noise_mode))
elif args.dataset== 'TinyImageNet':
    loader = dataloader(root=args.data_path, batch_size=args.batch_size, num_workers=4, ratio = args.ratio, noise_mode = args.noise_mode, noise_file='%s/clean_%.2f_%s.npz'%(args.data_path,args.ratio, args.noise_mode))


print('| Building net')
net = create_model()

# flow model
flowTrainer = FlowTrainer(args)
flowNet = flowTrainer.create_model()
flowTrainer.setEma(net, flowNet)

cudnn.benchmark = True

## Optimizer and Scheduler
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
optimizerFlow = optim.SGD(flowNet.parameters(), lr=args.lr_f, momentum=0.9, weight_decay=5e-4)
# optimizerFlow = optim.AdamW(flowNet.parameters(), lr=args.lr_f)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, args.lr / 100)
schedulerFlow = optim.lr_scheduler.CosineAnnealingLR(optimizerFlow, args.num_epochs, args.lr_f / 100)

## Resume from the warmup checkpoint 
model_name = 'Net_warmup.pth'
model_name_flow = 'FlowNet_warmup.pth'
model_name_ema = 'Net_warmup_ema.pth'
model_name_flow_ema = 'FlowNet_warmup_ema.pth'

if args.resume:
    start_epoch = args.warm_up
    net.load_state_dict(torch.load(os.path.join(model_save_loc, model_name))['net'])
    flowNet.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_flow))['net'])
    
    flowTrainer.net_ema.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_ema))['net'])
    flowTrainer.flowNet_ema.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_flow_ema))['net'])
    
elif args.pretrain != '':
    start_epoch = 0
    args.warm_up = 1
    net.load_state_dict(torch.load(args.pretrain)['net'])
else:
    start_epoch = 0

best_acc = 0
jsd_threshold = args.thr

## Warmup and SSL-Training 
for epoch in range(start_epoch,args.num_epochs+1):
    startTime = time.time() 
    test_loader = loader.run(0, 'val')
    eval_loader = loader.run(0, 'eval_train')   
    warmup_trainloader = loader.run(0,'warmup')
    
    ## Warmup Stage 
    if epoch<args.warm_up:       
        warmup_trainloader = loader.run(0, 'warmup')

        print('Warmup Model')
        flowTrainer.warmup_standard(epoch, net, flowNet, optimizer, optimizerFlow, warmup_trainloader)   
    
    else:
        ## Calculate JSD values and Filter Rate
        print("Calculate JSD")
        prob = flowTrainer.Calculate_JSD(net, flowNet, args.num_samples, eval_loader)
        SR , threshold = Selection_Rate(prob, jsd_threshold)
        if threshold.item()>args.d_u:
            threshold = threshold - (threshold-torch.min(prob))/args.tau
        jsd_threshold = threshold
        print('Train Net\n')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= prob) # Uniform Selection
        logJSD(epoch, threshold, labeled_trainloader, unlabeled_trainloader)
        flowTrainer.train(epoch, net, flowNet, optimizer, optimizerFlow, labeled_trainloader, unlabeled_trainloader)    # train net1  

    
    if args.w_ce:
        acc, confidence, acc_ce, confidence_ce, acc_mix, confidence_mix = flowTrainer.testByFlow(epoch, net, flowNet, test_loader)
    else:
        acc, confidence = flowTrainer.testByFlow(epoch, net, flowNet, test_loader)
    
    scheduler.step()
    schedulerFlow.step()

    ## wandb
    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["runtime"] = time.time() - startTime
        logMsg["acc/test"] = acc
        logMsg["confidence score"] = confidence
        if args.w_ce:
            logMsg["acc/test(ce_head)"] = acc_ce
            logMsg["confidence score(test_ce_head)"] = confidence_ce
            logMsg["acc/test(mix)"] = acc_mix
            logMsg["confidence score(mix)"] = confidence_mix
        wandb.log(logMsg)
    if acc > best_acc:

        save_model(net, flowNet, epoch, model_name, model_name_flow, acc)
        best_acc = acc

