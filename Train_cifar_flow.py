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
from flow_trainer import FlowTrainer
from flowModule.utils import standard_normal_logprob
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
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=1, type=float, help='weight for unsupervised loss')
parser.add_argument('--len_u', default=16, type=float, help='weight for unsupervised loss')
parser.add_argument('--warmup_u', default=10, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_x', default=30, type=float, help='weight for supervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=350, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--d_u',  default=0.47, type=float)
parser.add_argument('--tau', default=3.5, type=float, help='filtering coefficient')
parser.add_argument('--metric', type=str, default = 'JSD', help='Comparison Metric')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool, help = 'Resume from the warmup checkpoint')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data/cifar10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--flow_modules', default="8-8-8-8", type=str)
parser.add_argument('--name', default="", type=str)
parser.add_argument('--flowRefine', action='store_true')
parser.add_argument('--ce', action='store_true')
parser.add_argument('--fix', default='none', choices=['none', 'net', 'flow'], type=str)
parser.add_argument('--fix_wp', default='none', choices=['none', 'net', 'flow'], type=str)
parser.add_argument('--predictPolicy', default='mean', choices=['mean', 'weight'], type=str)
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
else:
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

## wandb
if (wandb != None):
    os.environ["WANDB_WATCH"] = "false"
    wandb.init(project="FlowUNICON", entity="andy-su", name=folder)
    wandb.config.update(args)
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("acc", summary="max")

# SSL-Training
def train(epoch, net, flownet, optimizer, optimizerFlow, labeled_trainloader, unlabeled_trainloader):
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
            features_u11, _ = net(inputs_u)
            features_u12, _ = net(inputs_u2)
                  
            
            ## Pseudo-label
            targets_u = flowTrainer.get_pseudo_label(flownet, features_u11, features_u12)
            # flow_outputs_u11 = flowTrainer.predict(flownet, features_u11)
            # flow_outputs_u12 = flowTrainer.predict(flownet, features_u12)

            # pu = (flow_outputs_u11 + flow_outputs_u12) / 2
            # ptu = pu**(1/args.T)            ## Temparature Sharpening
            
            # targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            # targets_u = targets_u.detach()                  

            ## Label refinement
            features_x, _  = net(inputs_x)
            features_x2, _ = net(inputs_x2)            
            
            px = flowTrainer.get_pseudo_label(flownet, features_x, features_x2)
            # flow_outputs_x = flowTrainer.predict(flownet, features_x)
            # flow_outputs_x2 = flowTrainer.predict(flownet, features_x2)
            # px = (flow_outputs_x + flow_outputs_x2) / 2   
            px = w_x*labels_x + (1-w_x)*px

            ptx = px**(1/args.T)    ## Temparature sharpening 
                        
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)           
            targets_x = targets_x.detach()
            # targets_x = labels_x

            # Calculate label sources
            u_sources_pesudo = Dist_Func(targets_u, labels_u_o)
            x_sources_origin = Dist_Func(labels_x, labels_x_o)
            x_sources_refine = Dist_Func(targets_x, labels_x_o)

        ## Unsupervised Contrastive Loss
        f1, _ = net(inputs_u3)
        f2, _ = net(inputs_u4)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss_simCLR = contrastive_criterion(features)

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
        all_inputs  = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b   = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        ## Mixup
        mixed_input  = l * input_a  + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        flow_feature, logits = net(mixed_input) # add flow_feature
        # logits_x = logits[:batch_size*2]
        # logits_u = logits[batch_size*2:]        
        
        ## Combined Loss
        # Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        ## Regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        ## Flow loss
        flow_feature = F.normalize(flow_feature, dim=1)
        flow_mixed_target = mixed_target.unsqueeze(1).cuda()
        delta_p = torch.zeros(flow_mixed_target.shape[0], flow_mixed_target.shape[1], 1).cuda() 
        approx21, delta_log_p2 = flownet(flow_mixed_target, flow_feature, delta_p)
        
        approx2 = standard_normal_logprob(approx21).view(mixed_target.size()[0], -1).sum(1, keepdim=True)
        delta_log_p2 = delta_log_p2.view(flow_mixed_target.size()[0], flow_mixed_target.shape[1], 1).sum(1)
        log_p2 = (approx2 - delta_log_p2)

        lamb_x = linear_rampup(epoch, warm_up, args.lambda_x) + 1
        lamb_u = linear_rampup(epoch, args.warmup_u, args.len_u, args.lambda_u)

        loss_nll_x = -log_p2[:batch_size*2].mean()
        loss_nll_u = -log_p2[batch_size*2:].mean()
        ## Total Loss
        loss = args.lambda_c * loss_simCLR + lamb_x * loss_nll_x + lamb_u * loss_nll_u #+ penalty

        # Compute gradient and Do SGD step
        optimizer.zero_grad()
        optimizerFlow.zero_grad()
        loss.backward()
        optimizer.step()
        optimizerFlow.step()

        ## wandb
        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["loss/nll_x"] = loss_nll_x.item()
            logMsg["loss/nll_u"] = loss_nll_u.item()
            logMsg["loss/simCLR"] = loss_simCLR.item()

            logMsg["label_quality/unlabel_pesudo_JSD_mean"] = u_sources_pesudo.mean().item()
            logMsg["label_quality/label_origin_JSD_mean"] = x_sources_origin.mean().item()
            logMsg["label_quality/label_refine_JSD_mena"] = x_sources_refine.mean().item()

            wandb.log(logMsg)

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Contrastive Loss:%.4f NLL(x) loss: %.2f NLL(u) loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss_simCLR.item(),  loss_nll_x.item(), loss_nll_u.item()))
        sys.stdout.flush()


## For Standard Training 
def warmup_standard(epoch, net, flownet, optimizer, optimizerFlow, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        optimizerFlow.zero_grad()
        feature, outputs = net(inputs)               
        #loss_ce = CEloss(outputs, labels)    

        # == flow ==
        feature = F.normalize(feature, dim=1)
        labels_one_hot = torch.nn.functional.one_hot(labels, args.num_class).type(torch.cuda.FloatTensor)
        flow_labels = labels_one_hot.unsqueeze(1).cuda()
        
        delta_p = torch.zeros(flow_labels.shape[0], flow_labels.shape[1], 1).cuda()
        approx21, delta_log_p2 = flownet(flow_labels, feature, delta_p)
        
        approx2 = standard_normal_logprob(approx21).view(flow_labels.size()[0], -1).sum(1, keepdim=True)
        delta_log_p2 = delta_log_p2.view(flow_labels.size()[0], flow_labels.shape[1], 1).sum(1)
        log_p2 = (approx2 - delta_log_p2)
        loss_nll = -log_p2.mean()
        # == flow end ===

        if args.noise_mode=='asym':     # Penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss_nll + penalty #+ loss_ce   
        else:   
            L = loss_nll #+ loss_ce

        L.backward()

        optimizer.step()  
        optimizerFlow.step()              

        ## wandb
        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["loss/nll"] = loss_nll.item()
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
    acc = flowTrainer.testByFlow(net, flowNet, test_loader)
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    
    ## wandb
    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["acc/test"] = acc
        wandb.log(logMsg)
    
    test_log.write(str(acc)+'\n')
    test_log.flush()  
    return acc


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

def Selection_Rate(prob):
    threshold = torch.mean(prob)
    if threshold.item()>args.d_u:
        threshold = threshold - (threshold-torch.min(prob))/args.tau
    SR = torch.sum(prob<threshold).item()/num_samples

    if SR <= 0 or SR >= 1.0:
        new_SR = np.clip(SR, 1/args.num_class, 0.9)
        print(f'WARNING: sample rate = {SR}, set to {new_SR}')
        SR = new_SR
    return SR, threshold

def Dist_Func(pred, target):
    JS_dist = Jensen_Shannon()
    dist = JS_dist(pred, F.one_hot(target, num_classes = args.num_class))
    return dist

## Calculate JSD
def Calculate_JSD(net, flowNet, num_samples):  
    JSD   = torch.zeros(num_samples)    

    for batch_idx, (inputs, targets, index) in tqdm(enumerate(eval_loader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            feature = net(inputs)[0]
            out = flowTrainer.predict(flowNet, feature)

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        dist = Dist_Func(out, targets)
        JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist

    return JSD

## Calculate JSD
def Calculate_JSD_onenet(model, num_samples):  
    JSD   = torch.zeros(num_samples)    

    for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            out = torch.nn.Softmax(dim=1).cuda()(model(inputs)[1])     

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

    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
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

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, args.lr / 100)
schedulerFlow = optim.lr_scheduler.CosineAnnealingLR(optimizerFlow, args.num_epochs, args.lr_f / 100)

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
else:
    start_epoch = 0

best_acc = 0

## Warmup and SSL-Training 
for epoch in range(start_epoch,args.num_epochs+1):   
    test_loader = loader.run(0, 'test')
    eval_loader = loader.run(0, 'eval_train')   
    warmup_trainloader = loader.run(0,'warmup')
    
    ## Warmup Stage 
    if epoch<warm_up:       
        warmup_trainloader = loader.run(0, 'warmup')

        print('Warmup Model')
        warmup_standard(epoch, net, flowNet, optimizer, optimizerFlow, warmup_trainloader)   
    
    else:
        ## Calculate JSD values and Filter Rate
        print("Calculate JSD")
        prob = Calculate_JSD(net, flowNet, num_samples)
        SR , threshold = Selection_Rate(prob)
        
        print('Train Net\n')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= prob) # Uniform Selection
        logJSD(epoch, threshold, labeled_trainloader, unlabeled_trainloader)
        train(epoch, net, flowNet,optimizer, optimizerFlow, labeled_trainloader, unlabeled_trainloader)    # train net1  

    acc = test(epoch,net, flowNet)
    scheduler.step()
    schedulerFlow.step()

    if acc > best_acc:
        if epoch <warm_up:
            model_name = 'Net_warmup.pth'
            model_name_flow = 'FlowNet_warmup.pth'
        else:
            model_name = 'Net.pth'
            model_name_flow = 'FlowNet.pth'

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
        best_acc = acc

