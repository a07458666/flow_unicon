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
from config import argumentParse
import numpy as np
from PreResNet_cifar import *
import dataloader_cifar as dataloader
from math import log2
from Contrastive_loss import *
import matplotlib.pyplot as plt

import time
import collections.abc
from collections.abc import MutableMapping
from flow_trainer import FlowTrainer
from flowModule.utils import standard_normal_logprob
from tqdm import tqdm

from torch_ema import ExponentialMovingAverage

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
if args.dataset== 'cifar10':
    torchvision.datasets.CIFAR10(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR10(args.data_path,train=False, download=True)
else:
    torchvision.datasets.CIFAR100(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR100(args.data_path,train=False, download=True)

## Checkpoint Location
folder = args.dataset + '_' + args.noise_mode + '_' + str(args.r)  + '_flow_' + args.name
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)
    os.mkdir(model_save_loc + '/JSD_distribution')
    os.mkdir(model_save_loc + '/NLL_distribution')

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
    wandb.init(project="FlowUNICON", entity="andy-su", name=folder)
    wandb.config.update(args)
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("acc", summary="max")

# SSL-Training
def train(epoch, net, flownet, net_ema, flowNet_ema, optimizer, optimizerFlow, labeled_trainloader, unlabeled_trainloader):
    net.train()
    flownet.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1

    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x, labels_x_o) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u_o = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u_o = unlabeled_train_iter.next()
        
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()
        
        labels_x_o = labels_x_o.cuda()
        labels_u_o = labels_u_o.cuda()

        with torch.no_grad():
            # Label co-guessing of unlabeled samples
            if args.ema:
                with net_ema.average_parameters():
                    features_u11, outputs_u11 = net(inputs_u)
                    features_u12, outputs_u12 = net(inputs_u2)
            else:
                features_u11, outputs_u11 = net(inputs_u)
                features_u12, outputs_u12 = net(inputs_u2)

            ## Pseudo-label
            pu_flow = flowTrainer.get_pseudo_label(flownet, features_u11, features_u12, std = args.pseudo_std)

            pu_net = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2
            log_pu(pu_flow, pu_net, labels_u_o)
            # pu_flow = pu_flow**(1/2)
            # pu = (pu_flow + pu_net) / 2
            pu = pu_flow
            
            lamb_Tu = (1 - linear_rampup(epoch+batch_idx/num_iter, args.warm_up, args.lambda_p, args.Tu))

            ptu = pu**(1/lamb_Tu)            ## Temparature Sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()                  

            ## Label refinement
            if args.ema:
                with net_ema.average_parameters():
                    features_x, _  = net(inputs_x)
                    features_x2, _ = net(inputs_x2)
            else:
                features_x, _  = net(inputs_x)
                features_x2, _ = net(inputs_x2)

            px_o = flowTrainer.get_pseudo_label(flownet, features_x, features_x2)
            # flow_outputs_x = flowTrainer.predict(flownet, features_x)
            # flow_outputs_x2 = flowTrainer.predict(flownet, features_x2)
            # px = (flow_outputs_x + flow_outputs_x2) / 2   
            px = w_x*labels_x + (1-w_x)*px_o

            ptx = px**(1/args.Tx)    ## Temparature sharpening 
                        
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)           
            targets_x = targets_x.detach()
            # print("targets_x : ", targets_x[:10])
            # print("targets_u : ", targets_u[:10])
            # targets_x = labels_x

            print_label_status(targets_x, targets_u, labels_x_o, labels_u_o, batch_idx)

            logFeature(torch.cat([features_u11, features_u12, features_x, features_x2], dim=0))
            # Calculate label sources
            u_sources_pseudo = Dist_Func(targets_u, labels_u_o)
            x_sources_origin = Dist_Func(labels_x, labels_x_o)
            x_sources_refine = Dist_Func(targets_x, labels_x_o)
            # print("x_sources_origin :", x_sources_origin)
            # print("labels_x :", labels_x[:20])
            # print("labels_x_o :", labels_x_o[:20])

        ## Unsupervised Contrastive Loss
        f1, _ = net(inputs_u3)
        f2, _ = net(inputs_u4)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss_simCLR = contrastive_criterion(features)


        all_inputs  = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        mixed_input, mixed_target = mix_match(all_inputs, all_targets, args.alpha)
                
        flow_feature, logits = net(mixed_input) # add flow_feature

        # Regularization feature var
        reg_f_var_loss = torch.clamp(1-torch.sqrt(flow_feature.var(dim=0) + 1e-10), min=0).mean()
        
        # logits_x = logits[:batch_size*2]
        # logits_u = logits[batch_size*2:]        
        
        ## Combined Loss
        # Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        ## Regularization
        # prior = torch.ones(args.num_class)/args.num_class
        # prior = prior.cuda()        
        # pred_mean = torch.softmax(logits, dim=1).mean(0)
        # penalty = torch.sum(prior*torch.log(prior/pred_mean))

        ## Flow loss
        flow_feature = F.normalize(flow_feature, dim=1)
        flow_mixed_target = mixed_target.unsqueeze(1).cuda()
        delta_p = torch.zeros(flow_mixed_target.shape[0], flow_mixed_target.shape[1], 1).cuda() 
        approx21, delta_log_p2 = flownet(flow_mixed_target, flow_feature, delta_p)
        
        approx2 = standard_normal_logprob(approx21).view(mixed_target.size()[0], -1).sum(1, keepdim=True)
        delta_log_p2 = delta_log_p2.view(flow_mixed_target.size()[0], flow_mixed_target.shape[1], 1).sum(1)
        log_p2 = (approx2 - delta_log_p2)

        # lamb_x = linear_rampup(epoch+batch_idx/num_iter, warm_up, args.linear_x, args.lambda_x) + 1
        lamb_u = linear_rampup(epoch+batch_idx/num_iter, args.warm_up, args.linear_u, args.lambda_u) + 1
        
        loss_nll_x = -log_p2[:batch_size*2]
        loss_nll_u = -log_p2[batch_size*2:]

        # log_loss(loss_nll_x.cpu().detach().numpy(), loss_nll_u.cpu().detach().numpy(), flow_mixed_target[:batch_size*2].cpu().detach().numpy(), flow_mixed_target[batch_size*2:].cpu().detach().numpy(), labels_x.cpu().detach().numpy(), labels_x_o.cpu().detach().numpy(), batch_idx)
        # mixup_x = mixed_target[:batch_size*2]
        # mixup_u = mixed_target[batch_size*2:]
        # print("mixed_target_x : ", mixup_x[:10])
        # print("mixed_target_u : ", mixup_u[:10])
        # print("loss_nll_x : ", loss_nll_x[:10])
        # print("loss_nll_u : ", loss_nll_u[:10])
        # print("targets_x : ", targets_x[:10])
        # print("px_o : ", px_o[:10])
        # print("targets_u : ", targets_u[:10])
        # print("labels_x_o : ", labels_x_o[:10])
        # print("labels_u_o : ", labels_u_o[:10])

        # print("loss_CLR: ", args.lambda_c * loss_simCLR )
        ## Total Loss
        loss = args.lambda_c * loss_simCLR + reg_f_var_loss + (-log_p2).mean() #loss_nll_x.mean() + lamb_u * loss_nll_u.mean() #+ penalty #  Lx + lamb * Lu 
        if args.clip_grad:
            loss = clipping_grad(loss)
        
        # Compute gradient and Do SGD step
        optimizer.zero_grad()
        optimizerFlow.zero_grad()
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_value_(net.parameters(), 4)
            torch.nn.utils.clip_grad_value_(flownet.parameters(), 4)
        if args.fix == 'flow':
            optimizer.step()
        elif args.fix == 'net':
            optimizerFlow.step()  
        else:
            optimizer.step()
            optimizerFlow.step()  

        if args.ema:
            net_ema.update()
            flowNet_ema.update()

        # model_name = 'Net_{}.pth'.format(batch_idx)
        # model_name_flow = 'FlowNet_{}.pth'.format(batch_idx)
        # save_model(net, flowNet, epoch, model_name, model_name_flow)

        ## wandb
        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["lamb_Tu"] = lamb_Tu
            
            logMsg["loss/nll_x"] = loss_nll_x.mean().item()
            logMsg["loss/nll_u"] = loss_nll_u.mean().item()

            logMsg["loss/nll_x_max"] = loss_nll_x.max()
            logMsg["loss/nll_x_min"] = loss_nll_x.min()
            logMsg["loss/nll_x_var"] = loss_nll_x.var()
            logMsg["loss/nll_u_max"] = loss_nll_u.max()
            logMsg["loss/nll_u_min"] = loss_nll_u.min()
            logMsg["loss/nll_u_var"] = loss_nll_u.var()

            logMsg["loss/simCLR"] = loss_simCLR.item()

            logMsg["label_quality/unlabel_pseudo_JSD_mean"] = u_sources_pseudo.mean().item()
            logMsg["label_quality/label_origin_JSD_mean"] = x_sources_origin.mean().item()
            logMsg["label_quality/label_refine_JSD_mena"] = x_sources_refine.mean().item()

            wandb.log(logMsg)

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Contrastive Loss:%.4f NLL(x) loss: %.2f NLL(u) loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss_simCLR.item(),  loss_nll_x.mean().item(), loss_nll_u.mean().item()))
        sys.stdout.flush()


## For Standard Training 
def warmup_standard(epoch, net, flownet, net_ema, flowNet_ema, optimizer, optimizerFlow, dataloader):
    flownet.train()
    if args.fix == 'net':
        net.eval()
    else:    
        net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        labels_one_hot = torch.nn.functional.one_hot(labels, args.num_class).type(torch.cuda.FloatTensor)

        if args.warmup_mixup:
            mixed_input, mixed_target = mix_match(inputs, labels_one_hot, args.alpha_warmup)
            feature, outputs = net(mixed_input)
            flow_labels = mixed_target.unsqueeze(1).cuda()
        else:  
            feature, outputs = net(inputs)
            flow_labels = labels_one_hot.unsqueeze(1).cuda()
        # mixed_target = distillation_label(labels_one_hot, outputs)

        logFeature(feature)            
        # loss_ce = CEloss(outputs, labels)    

        # == flow ==
        feature = F.normalize(feature, dim=1)

        delta_p = torch.zeros(flow_labels.shape[0], flow_labels.shape[1], 1).cuda()
        approx21, delta_log_p2 = flownet(flow_labels, feature, delta_p)
        
        approx2 = standard_normal_logprob(approx21).view(flow_labels.size()[0], -1).sum(1, keepdim=True)
        delta_log_p2 = delta_log_p2.view(flow_labels.size()[0], flow_labels.shape[1], 1).sum(1)
        log_p2 = (approx2 - delta_log_p2)
        loss_nll = -log_p2.mean()
        # == flow end ===

        if args.noise_mode=='asym':     # Penalize confident prediction for asymmetric noise
            #penalty = conf_penalty(outputs)
            L = loss_nll #+ penalty #+ loss_ce   
        else:   
            L = loss_nll #+ loss_ce


        optimizer.zero_grad()
        optimizerFlow.zero_grad()
        L.backward()

        if args.fix == 'flow':
            optimizer.step()
        elif args.fix == 'net':
            optimizerFlow.step()   
        else:
            optimizer.step()
            optimizerFlow.step()  

        if args.ema:
            net_ema.update()
            flowNet_ema.update()

        ## wandb
        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["loss/nll"] = loss_nll.item()
            logMsg["loss/nll_max"] = (-log_p2).max()
            logMsg["loss/nll_min"] = (-log_p2).min()
            logMsg["loss/nll_var"] = (-log_p2).var()
            wandb.log(logMsg)

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t NLL-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss_nll.item()))
        sys.stdout.flush()

## For Training Accuracy
def warmup_val(epoch,net,optimizer, optimizerFlow,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    total = 0
    correct = 0
    loss_x = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
            inputs, labels = inputs.cuda(), labels.cuda() 
            optimizer.zero_grad()
            optimizerFlow.zero_grad()
            _, outputs  = net(inputs)

            _, predicted = torch.max(outputs, 1)    
            # loss    = CEloss(outputs, labels)    
            # loss_x += loss.item()                      

            total   += labels.size(0)
            correct += predicted.eq(labels).cpu().sum().item()

    acc = 100.*correct/total
    print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch, acc))  
    
    ## wandb
    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["acc/warmup"] = acc
        wandb.log(logMsg)

    train_loss.write(str(loss_x/(batch_idx+1)))
    train_acc.write(str(acc))
    train_acc.flush()
    train_loss.flush()

    return acc

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

# KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

def Selection_Rate(prob, pre_threshold):
    threshold = torch.mean(prob)
    if args.ema_jsd:
        threshold = (args.jsd_decay * pre_threshold) + ((1 - args.jsd_decay) * threshold)
    print("threshold : ", torch.mean(prob))
    print("threshold(new) : ", threshold)
    print("prob size : ", prob.size())
    print("down :", torch.sum(prob<threshold).item())
    print("up :", torch.sum(prob>threshold).item())
    SR = torch.sum(prob<threshold).item()/num_samples
    print("SR :", SR)
    if SR <= 0.1 or SR >= 0.9:
        new_SR = np.clip(SR, 0.1, 0.9)
        print(f'WARNING: sample rate = {SR}, set to {new_SR}')
        SR = new_SR
    return SR, threshold

def Dist_Func(pred, target):
    JS_dist = Jensen_Shannon()
    dist = JS_dist(pred, F.one_hot(target, num_classes = args.num_class))
    return dist

D_GRAD_CLIP = 1e14
def clipping_grad(loss):
    if loss.requires_grad:
        def hook(grad):
            if (wandb != None):
                logMsg = {}
                logMsg["epoch"] = epoch
                logMsg["max_grads"] =  float(grad.abs().max().cpu().numpy())
                wandb.log(logMsg)
            clipped_grad = grad.clamp(min=-D_GRAD_CLIP, max=D_GRAD_CLIP)
            return clipped_grad
        loss.register_hook(hook)
    return loss

## Calculate JSD
def Calculate_JSD(net, flowNet, num_samples):  
    JSD   = torch.zeros(num_samples)    
    for batch_idx, (inputs, targets, index) in tqdm(enumerate(eval_loader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            if args.ema:
                with net_ema.average_parameters():
                    feature = net(inputs)[0]
            else:
                feature = net(inputs)[0]
            feature = F.normalize(feature, dim=1)
            out = flowTrainer.predict(flowNet, feature)

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        dist = Dist_Func(out, targets)
        JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist

    return JSD

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

    # plt.clf()
    # kwargs = dict(histtype='stepfilled', alpha=0.75, density=False, bins=20)
    # plt.hist(clean_prob, color='green', range=(0., 1.), label='clean', **kwargs)
    # plt.hist(noise_prob, color='red'  , range=(0., 1.), label='noisy', **kwargs)

    # plt.xlabel('NLL Values')
    # plt.ylabel('count')
    # plt.title(f'NLL Distribution of N Samples epoch :{epoch}')
    # plt.grid(True)
    # plt.savefig(f'{model_save_loc}/NLL_distribution/epoch{epoch}.png')

    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["JSD"] = wandb.Image(f'{model_save_loc}/JSD_distribution/epoch{epoch}.png')
        # logMsg["NLL"] = wandb.Image(f'{model_save_loc}/NLL_distribution/epoch{epoch}.png')
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

## Unsupervised Loss coefficient adjustment 
def linear_rampup(current, warm_up, rampup_length=16, lambda_w=1.0):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return lambda_w * float(current)

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

def distillation_label(labels, pseudo, alpha = 0.2):
    with torch.no_grad():
        l = np.random.beta(alpha, alpha)        
        l = max(l, 1-l)
        new_label = (l * labels) + ((1 - l) * pseudo.detach())
        return new_label.detach()


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


def log_loss(loss_nll_x, loss_nll_u, target_x, target_u, labels_x, labels_x_o, batch_idx):
    count = 10
    print("loss_nll_x : ", loss_nll_x[:count])
    print("loss_nll_u : ", loss_nll_u[:count])
    print("target_x : ", target_x[:count])
    print("target_u : ", target_u[:count])
    print("labels_x : ", labels_x[:count])
    print("labels_x_o : ", labels_x_o[:count])


    loss_log.write('\n epoch : ' + str(epoch))
    loss_log.write("\n loss_nll_x size : " + str(loss_nll_x.shape[0]))
    loss_log.write("\n loss_nll_u size : " + str(loss_nll_u.shape[0]))
    loss_log.write('\n loss_nll_x : ' + str(loss_nll_x[:count]))
    loss_log.write('\n loss_nll_u : ' + str(loss_nll_u[:count]))
    loss_log.write('\n target_x : ' + str(target_x[:count]))
    loss_log.write('\n target_u : ' + str(target_u[:count]))
    loss_log.write('\n labels_x_o : ' + str(labels_x_o[:count]))
    
    loss_log.flush()

    plt.clf()
    
    kwargs = dict(histtype='stepfilled', alpha=0.75, density=False, bins=40)
    plt.hist(loss_nll_x, color='green', label='loss_x', **kwargs)
    plt.hist(loss_nll_u, color='red'  , label='loss_u', **kwargs)

    plt.xlabel('NLL Values')
    plt.ylabel('count')
    plt.title(f'NLL Distribution of N Samples epoch :{epoch}_{batch_idx}')
    plt.grid(True)
    plt.savefig(f'{model_save_loc}/NLL_distribution/epoch{epoch}_{batch_idx}.png')

    # prob_flow, predicted_flow = torch.max(pu_flow, 1)
    # prob_net, predicted_net = torch.max(pu_net, 1)

    # total = gt.size(0)
    # correct_flow = predicted_flow.eq(gt).cpu().sum().item()  
    # prob_sum_flow = prob_flow.cpu().sum().item()

    # correct_net = predicted_net.eq(gt).cpu().sum().item()  
    # prob_sum_net = prob_net.cpu().sum().item()

    # acc_flow = 100.*correct_flow/total
    # confidence_flow = prob_sum_flow/total

    # acc_net = 100.*correct_net/total
    # confidence_net = prob_sum_net/total

    # pu_log.write('\n epoch : ' + str(epoch))
    # pu_log.write('\n acc_flow : ' + str(acc_flow))
    # pu_log.write('\n confidence_flow : ' + str(confidence_flow))
    
    # pu_log.write('\n acc_net : ' + str(acc_net))
    # pu_log.write('\n confidence_net : ' + str(confidence_net))

    # pu_log.write('\n pu_flow : ' + str(pu_flow[:5].cpu().numpy()))
    # pu_log.write('\n pu_net : ' + str(pu_net[:5].cpu().numpy()))
    # pu_log.write('\n gt : ' + str(gt[:5].cpu().numpy()))
    
    # pu_log.flush()

    # if (wandb != None):
    #     logMsg = {}
    #     logMsg["epoch"] = epoch
    #     logMsg["loss_di/acc_flow"] = acc_flow
    #     logMsg["pseudo/confidence_flow"] = confidence_flow
    #     logMsg["pseudo/acc_net"] = acc_net
    #     logMsg["pseudo/confidence_net"] = confidence_net
    #     wandb.log(logMsg)
    return

def data2Tab(labels, values, name):
    data = [[label, val] for (label, val) in zip(labels, values)]
    table = wandb.Table(data=data, columns = ["label", "value"])
    return wandb.plot.bar(table, "label", "value", title=name)

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
    # print(f"\n=====epoch{epoch}======")
    # print("pseudo_labels_u : ", pseudo_labels_u)
    # print("target_labels_u : ", target_labels_u)
    # print("refine_labels_x : ", refine_labels_x)
    # print("target_labels_x : ", target_labels_x)

    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["label_count/pseudo_labels_u"] =  max(pseudo_labels_u)
        logMsg["label_count/target_labels_u"] =  max(target_labels_u)
        logMsg["label_count/refine_labels_x"] =  max(refine_labels_x)
        logMsg["label_count/target_labels_x"] =  max(target_labels_x)
        wandb.log(logMsg)

def mix_match(inputs, targets, alpha = 4):
    # MixMatch
    l = np.random.beta(alpha, alpha)        
    l = max(l, 1-l)

    idx = torch.randperm(inputs.size(0))

    input_a, input_b   = inputs, inputs[idx]
    target_a, target_b = targets, targets[idx]
    
    ## Mixup
    mixed_input  = l * input_a  + (1 - l) * input_b        
    mixed_target = l * target_a + (1 - l) * target_b
    return mixed_input, mixed_target

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
flowTrainer = FlowTrainer(args)
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

## Loss Functions
CE       = nn.CrossEntropyLoss(reduction='none')
CEloss   = nn.CrossEntropyLoss()
MSE_loss = nn.MSELoss(reduction= 'none')
contrastive_criterion = SupConLoss()
KLloss = nn.KLDivLoss(reduction="batchmean")

if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

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

jsd_threshold = args.thr

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
        SR , threshold = Selection_Rate(prob, jsd_threshold)
        jsd_threshold = threshold
        print('Train Net\n')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= prob) # Uniform Selection
        logJSD(epoch, threshold, labeled_trainloader, unlabeled_trainloader)
        train(epoch, net, flowNet, net_ema, flowNet_ema, optimizer, optimizerFlow, labeled_trainloader, unlabeled_trainloader)    # train net1  

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

    # model_name = 'Net_{}.pth'.format(epoch)
    # model_name_flow = 'FlowNet_{}.pth'.format(epoch)
    # save_model(net, flowNet, epoch, model_name, model_name_flow, acc)

