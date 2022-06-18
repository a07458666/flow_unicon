from __future__ import print_function
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

import collections.abc
from collections.abc import MutableMapping
from flow_trainer import FlowTrainer
from flowModule.utils import standard_normal_logprob

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
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=30, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=350, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--d_u',  default=0.7, type=float)
parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
parser.add_argument('--metric', type=str, default = 'JSD', help='Comparison Metric')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool, help = 'Resume from the warmup checkpoint')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data/cifar10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--flow_modules', default="8-8-8-8", type=str)
parser.add_argument('--name', default="", type=str)
parser.add_argument('--flowPseudo', action='store_true')
parser.add_argument('--flowRefine', action='store_true')
parser.add_argument('--oneNet', action='store_true')
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
def train(epoch, net, net2, flownet, optimizer, labeled_trainloader, unlabeled_trainloader):
    net2.eval() # Freeze one network and train the other
    net.train()
    flownet.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1

    ## Loss statistics
    loss_x = 0
    loss_u = 0
    loss_scl = 0
    loss_ucl = 0
    loss_nll = 0

    kld = 0

    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
        
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()
        
        with torch.no_grad():
            # Label co-guessing of unlabeled samples
            features_u11, outputs_u11 = net(inputs_u)
            features_u12, outputs_u12 = net(inputs_u2)
            _, outputs_u21 = net2(inputs_u)
            _, outputs_u22 = net2(inputs_u2)            
            
            ## Pseudo-label
            if (args.flowPseudo):
                flow_outputs_u11 = flowTrainer.predict(flownet, features_u11)
                flow_outputs_u12 = flowTrainer.predict(flownet, features_u12)
                
                pu_f = (torch.softmax(flow_outputs_u11, dim=1) + torch.softmax(flow_outputs_u12, dim=1)) / 2
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
                kld += KLloss(pu_f, pu).item()
            else:
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       

            ptu = pu**(1/args.T)            ## Temparature Sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()                  

            ## Label refinement
            _, outputs_x  = net(inputs_x)
            _, outputs_x2 = net(inputs_x2)            
            
            if (args.flowRefine):
                px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2    
                px = w_x*labels_x + (1-w_x)*px
            else:
                px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                px = w_x*labels_x + (1-w_x)*px      

            ptx = px**(1/args.T)    ## Temparature sharpening 
                        
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)           
            targets_x = targets_x.detach()

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
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
        
        ## Combined Loss
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
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
        loss_flow_nll = -log_p2.mean()

        ## Total Loss
        loss = Lx + lamb * Lu + args.lambda_c*loss_simCLR + penalty + loss_flow_nll

        ## Accumulate Loss
        loss_x += Lx.item()
        loss_u += Lu.item()
        loss_ucl += loss_simCLR.item()
        loss_nll += loss_flow_nll.item()

        # Compute gradient and Do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        ## wandb
        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["loss/loss_nll"] = loss_nll/(batch_idx+1)
            logMsg["loss/loss_x"] = loss_x/(batch_idx+1)
            logMsg["loss/loss_u"] = loss_u/(batch_idx+1)
            logMsg["loss/loss_ucl"] = loss_ucl/(batch_idx+1)
            logMsg["kld"] = kld / (batch_idx+1)
            wandb.log(logMsg)

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f Contrastive Loss:%.4f NLL loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss_x/(batch_idx+1), loss_u/(batch_idx+1),  loss_ucl/(batch_idx+1),  loss_nll/(batch_idx+1)))
        sys.stdout.flush()


## For Standard Training 
def warmup_standard(epoch,net,flownet, optimizer,dataloader):

    loss_nll_t = 0
    loss_ce_t = 0

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        feature, outputs = net(inputs)               
        loss_ce    = CEloss(outputs, labels)    

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
        loss = loss_nll + loss_ce

        if args.noise_mode=='asym':     # Penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        else:   
            L = loss

        loss_nll_t += loss_nll.item()
        loss_ce_t += loss_ce.item()
        L.backward()  
        optimizer.step()                

        ## wandb
        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["loss/nll+ce"] = (loss_nll_t + loss_ce_t) / (batch_idx + 1)
            logMsg["loss/nll"] = loss_nll_t / (batch_idx + 1)
            logMsg["loss/ce"] = loss_ce_t / (batch_idx + 1)
            wandb.log(logMsg)

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f NLL-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss_ce_t / (batch_idx + 1), loss_nll_t / (batch_idx + 1)))
        sys.stdout.flush()

## For Training Accuracy
def warmup_val(epoch,net,optimizer,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    total = 0
    correct = 0
    loss_x = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
            inputs, labels = inputs.cuda(), labels.cuda() 
            optimizer.zero_grad()
            _, outputs  = net(inputs)               
            _, predicted = torch.max(outputs, 1)    
            loss    = CEloss(outputs, labels)    
            loss_x += loss.item()                      

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
def test(epoch,net1,net2):
    net1.eval()
    net2.eval()

    num_samples = 1000
    correct = 0
    total = 0
    loss_x = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1 = net1(inputs)
            _, outputs2 = net2(inputs)
            if (args.oneNet):      
                outputs = outputs1
            else:
                outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
            loss = CEloss(outputs, targets)  
            loss_x += loss.item()

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()  

    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    
    ## wandb
    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["acc/test"] = acc
        wandb.log(logMsg)
    
    test_log.write(str(acc)+'\n')
    test_log.flush()  
    test_loss_log.write(str(loss_x/(batch_idx+1))+'\n')
    test_loss_log.flush()
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

## Calculate JSD
def Calculate_JSD(model1, model2, num_samples):  
    JS_dist = Jensen_Shannon()
    JSD   = torch.zeros(num_samples)    

    for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])     
            out2 = torch.nn.Softmax(dim=1).cuda()(model2(inputs)[1])
        ## Get the Prediction
        out = (out1 + out2)/2     

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        dist = JS_dist(out,  F.one_hot(targets, num_classes = args.num_class))
        JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist

    return JSD


## Unsupervised Loss coefficient adjustment 
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch,warm_up)

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
net1 = create_model()
net2 = create_model()

# flow model
flowTrainer = FlowTrainer(args)
flowNet1 = flowTrainer.create_model()
flowNet2 = flowTrainer.create_model()

cudnn.benchmark = True

## Semi-Supervised Loss
criterion  = SemiLoss()

## Optimizer and Scheduler
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 280, 2e-4)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, 280, 2e-4)

## Loss Functions
CE       = nn.CrossEntropyLoss(reduction='none')
CEloss   = nn.CrossEntropyLoss()
MSE_loss = nn.MSELoss(reduction= 'none')
contrastive_criterion = SupConLoss()
KLloss = nn.KLDivLoss(reduction="batchmean")

if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

## Resume from the warmup checkpoint 
model_name_1 = 'Net1_warmup.pth'
model_name_2 = 'Net2_warmup.pth'

model_name_flow_1 = 'FlowNet1_warmup.pth'
model_name_flow_2 = 'FlowNet2_warmup.pth'

if args.resume:
    start_epoch = warm_up
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1))['net'])
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_2))['net'])
    flowNet1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_flow_1))['net'])
    flowNet2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_flow_2))['net'])
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

        print('Warmup Model 1')
        warmup_standard(epoch, net1, flowNet1, optimizer1, warmup_trainloader)   
        print('\nWarmup Model 2')
        warmup_standard(epoch, net2, flowNet2, optimizer2, warmup_trainloader) 
    
    else:
        if (args.metric == "JSD"):
            ## Calculate JSD values and Filter Rate
            print("Calculate JSD net 1")
            prob = Calculate_JSD(net2, net1, num_samples)
            print("prob s", prob.size())
            threshold = torch.mean(prob)
            if threshold.item()>args.d_u:
                threshold = threshold - (threshold-torch.min(prob))/args.tau
            selectMetric = prob
            SR = torch.sum(prob<threshold).item()/num_samples    

        elif (args.metric == "density"):
            ## Calculate density(flow) values and Filter Rate
            print("Calculate density net 1")
            densitys = flowTrainer.Calculate_density(net1, net2, flowNet1, flowNet2, num_samples, eval_loader)
            print("densitys size", densitys.size())
            print("densitys ", densitys[:20])
            threshold = torch.mean(densitys)
            if threshold.item() > torch.max(densitys) * args.d_u:
                threshold = threshold - (threshold-torch.min(densitys))/args.tau
            SR = torch.sum(densitys<threshold).item()/num_samples    
            selectMetric = densitys
            print("SR : ", SR)
            print("threshold ", threshold)
        
        print('Train Net1\n')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= selectMetric) # Uniform Selection
        train(epoch,net1,net2,flowNet1,optimizer1,labeled_trainloader, unlabeled_trainloader)    # train net1  
        
         # only train net1
        if (args.oneNet):
            continue

        if (args.metric == "density"):
            ## Calculate JSD values and Filter Rate
            print("Calculate JSD net 2")
            prob = Calculate_JSD(net2, net1, num_samples)      
            print("prob s", prob.size())     
            threshold = torch.mean(prob)
            if threshold.item()>args.d_u:
                threshold = threshold - (threshold-torch.min(prob))/args.tau
            selectMetric = prob
            SR = torch.sum(prob<threshold).item()/num_samples
        elif (args.metric == "density"):
            ## Calculate density(flow) values and Filter Rate
            print("Calculate density net 2")
            densitys = flowTrainer.Calculate_density(net1, net2, flowNet1, flowNet2, num_samples, eval_loader)
            print("densitys 2 size", densitys.size())
            print("densitys 2 ", densitys[:20])
            threshold = torch.mean(densitys)
            if threshold.item() > torch.max(densitys) * args.d_u:
                threshold = threshold - (threshold-torch.min(densitys))/args.tau
            SR = torch.sum(densitys<threshold).item()/num_samples  
            selectMetric = densitys      
            print("SR 2: ", SR)
            print("threshold 2 ", threshold)
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= selectMetric)     # Uniform Selection
        train(epoch, net2,net1,flowNet2, optimizer2,labeled_trainloader, unlabeled_trainloader)       # train net2

    acc = test(epoch,net1,net2)
    scheduler1.step()
    scheduler2.step()

    if acc > best_acc:
        if epoch <warm_up:
            model_name_1 = 'Net1_warmup.pth'
            model_name_2 = 'Net2_warmup.pth'
            model_name_flow_1 = 'FlowNet1_warmup.pth'
            model_name_flow_2 = 'FlowNet2_warmup.pth'
        else:
            model_name_1 = 'Net1.pth'
            model_name_2 = 'Net2.pth'
            model_name_flow_1 = 'FlowNet1.pth'
            model_name_flow_2 = 'FlowNet2.pth'

        print("Save the Model-----")
        checkpoint1 = {
            'net': net1.state_dict(),
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

        checkpoint2 = {
            'net': net2.state_dict(),
            'Model_number': 2,
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

        checkpoint_flow1 = {
            'net': flowNet1.state_dict(),
            'Model_number': 3,
            'Noise_Ratio': args.r,
            'Loss Function': 'log-likelihood',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Dataset': 'TinyImageNet',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        checkpoint_flow2 = {
            'net': flowNet2.state_dict(),
            'Model_number': 4,
            'Noise_Ratio': args.r,
            'Loss Function': 'log-likelihood',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Dataset': 'TinyImageNet',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
        torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))
        torch.save(checkpoint_flow1, os.path.join(model_save_loc, model_name_flow_1))
        torch.save(checkpoint_flow2, os.path.join(model_save_loc, model_name_flow_2))
        best_acc = acc

