from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
import argparse
import numpy as np
import dataloader_clothing1M as dataloader
from torch_ema import ExponentialMovingAverage
import time
from sklearn.mixture import GaussianMixture
import copy 
import torchnet
from Contrastive_loss import *
from PreResNet_clothing1M import *
import matplotlib.pyplot as plt

# flow
from flow_trainer import FlowTrainer
from flowModule.utils import standard_normal_logprob, linear_rampup, mix_match
from flowModule.jensen_shannon import js_distance


try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")


parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')   ## Set the learning rate to 0.005 for faster training at the beginning
parser.add_argument('--lr_f', default=2e-5, type=float, help='initial flow learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--data_path', default='./data/Clothing1M_org', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default="0", help='comma separated list of GPU(s) to use.')
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=1000, type=int)
parser.add_argument('--dataset', default="Clothing1M", type=str)
parser.add_argument('--resume', default=False, type=bool, help = 'Resume from the warmup checkpoint')
parser.add_argument('--warm_up', default=0, type=int)
parser.add_argument('--name', default="", type=str)
parser.add_argument('--flow_modules', default="160-160-160-160", type=str)
parser.add_argument('--tol', default=1e-5, type=float, help='flow atol, rtol')
parser.add_argument('--ema_decay', default=0.99, type=float, help='ema decay')
parser.add_argument('--cond_size', default=512, type=int)
parser.add_argument('--lambda_f', default=1., type=float, help='flow nll loss weight')
parser.add_argument('--pseudo_std', default=0.2, type=float)

parser.add_argument('--centering', default=True, type=bool, help='use centering')
parser.add_argument('--lossType', default='nll', type=str, choices=['nll', 'ce', 'mix'], help = 'useing nll, ce or nll + ce loss')
parser.add_argument('--clip_grad', default=False, help = 'cliping grad')
parser.add_argument('--d_u',  default=0.7, type=float)

# Flow Sharpening
parser.add_argument('--flow_sp', default=False, type=bool, help='flow sharpening')
parser.add_argument('--lambda_p', default=50, type=float, help='sharpening lamb')
parser.add_argument('--Tf_warmup', default=1.0, type=float, help='warm-up flow sharpening temperature')
parser.add_argument('--Tf', default=0.7, type=float, help='flow sharpening temperature')
parser.add_argument('--sharpening', default="UNICON", type=str, choices=['DINO', 'UNICON'], help = 'sharpening method')
# Flow Centering
parser.add_argument('--center_momentum', default=0.8, type=float, help='use centering')

# Flow w_u
parser.add_argument('--linear_u', default=16, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_flow_u', default=1, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_flow_u_warmup', default=0.1, type=float, help='weight for unsupervised loss start value')

parser.add_argument('--lambda_u', default=30, type=float, help='weight for unsupervised loss')
args = parser.parse_args()

# torch.cuda.set_device(args.gpuid)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
contrastive_criterion = SupConLoss()

## For plotting the logs
# import wandb
# wandb.init(project="noisy-label-project-clothing1M", entity="...")

## Training
def train(epoch, net1, flowNet1, net2, flowNet2, optimizer, optimizer_flow, labeled_trainloader, unlabeled_trainloader):
    net1.train()         # Fix one network and train the other    
    flowNet1.train()
    net2.eval()
    flowNet2.eval()
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    
    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)
        
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()
        
        lamb_Tu = linear_rampup(epoch+batch_idx/num_iter, args.warm_up, args.lambda_p, args.Tf_warmup, args.Tf)

        with torch.no_grad():
            pu_net_1, pu_flow_1 = flowTrainer.get_pseudo_label(net1, flowNet1, inputs_u, inputs_u2, std = args.pseudo_std)
            pu_net_2, pu_flow_2 = flowTrainer.get_pseudo_label(net2, flowNet2, inputs_u, inputs_u2, std = args.pseudo_std)
  
            pu_net_sp_1 = flowTrainer.sharpening(pu_net_1, args.T)
            pu_net_sp_2 = flowTrainer.sharpening(pu_net_2, args.T)
            
            if args.flow_sp:
                pu_flow_1 = flowTrainer.sharpening(pu_flow_1, lamb_Tu)
                pu_flow_2 = flowTrainer.sharpening(pu_flow_2, lamb_Tu)

            ## Pseudo-label
            if args.lossType == "ce":
                targets_u = (pu_net_sp_1 + pu_net_sp_2) / 2
            elif args.lossType == "nll":
                targets_u = (pu_flow_1 + pu_flow_2) / 2
            elif args.lossType == "mix":
                targets_u = (pu_net_sp_1 + pu_net_sp_2 + pu_flow_1 + pu_flow_2) / 4

            targets_u = targets_u.detach()

            px_net_1, px_flow_1 = flowTrainer.get_pseudo_label(net1, flowNet1, inputs_x, inputs_x2, std = args.pseudo_std)

            if args.lossType == "ce":
                px = px_net_1
            elif args.lossType == "nll":
                px = px_flow_1
            elif args.lossType == "mix":
                px = (px_net_1 + px_flow_1) / 2

            px_mix = w_x*labels_x + (1-w_x)*px

            targets_x = flowTrainer.sharpening(px_mix, args.T)        
            targets_x = targets_x.detach()

            ## updateCnetering
            if args.centering:
                _, _ = flowTrainer.get_pseudo_label(net1, flowNet1, inputs_u, inputs_u2, std = args.pseudo_std, updateCnetering = True)   

        ## Unsupervised Contrastive Loss
        f1, _ = net1(inputs_u3)
        f2, _ = net1(inputs_u4)

        features    = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss_simCLR = contrastive_criterion(features)

        all_inputs  = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        mixed_input, mixed_target = mix_match(all_inputs, all_targets, args.alpha) 

        _, logits, flow_feature = net1(mixed_input, get_feature = True)

        # ## Test Lu loss
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
        
        # ## Semi-supervised Loss
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2],
                                logits_u, mixed_target[batch_size*2:],
                                epoch+batch_idx/num_iter, warm_up,
                                args.lambda_u, args.linear_u)
        # Lx = -torch.mean(
        #     torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1)
        # )
        ## Regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss_ce = Lx + lamb * Lu + penalty

        ## Flow
        _, log_p2 = flowTrainer.log_prob(mixed_target.unsqueeze(1).cuda(), flow_feature, flowNet1)

        lamb_u = linear_rampup(epoch+batch_idx/num_iter, args.warm_up, args.linear_u, args.lambda_flow_u_warmup, args.lambda_flow_u)   

        loss_nll_x = -log_p2[:batch_size*2]
        loss_nll_u = -log_p2[batch_size*2:]
        
        log_p2[batch_size*2:] *= lamb_u
        loss_nll = (-log_p2).mean()
        
        loss_flow = (args.lambda_f * loss_nll)
        
        ## Total Loss
        if args.lossType == "mix":
            loss = loss_flow + loss_ce + (args.lambda_c * loss_simCLR)
        elif args.lossType == "ce":
            loss = loss_ce + (args.lambda_c * loss_simCLR)
        elif args.lossType == "nll":
            loss = loss_flow + (args.lambda_c * loss_simCLR)

        ## Compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_flow.zero_grad()
        loss.backward()
        if args.clip_grad:
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 1e-10)
                torch.nn.utils.clip_grad_norm_(flownet.parameters(), 1e-10)
        optimizer.step()
        optimizer_flow.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s: | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE loss: %.2f  Contrative Loss:%.4f nll Loss:%.4f'
                %(args.dataset,  epoch, args.num_epochs, batch_idx+1, num_iter, loss_ce, loss_simCLR, loss_flow))
        sys.stdout.flush()

        ## wandb
        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["loss/loss_ce"] = loss_ce.item()
            logMsg["loss/loss_simCLR"] = loss_simCLR.item()
            logMsg["loss/penalty"] = penalty.item()
            logMsg["loss/loss_flow"] = loss_flow.item()

def warmup(net, flowNet, optimizer, optimizer_flow, dataloader):
    net.train()
    flowNet.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        optimizer_flow.zero_grad()
        _ , outputs, feature_flow = net(inputs, get_feature = True)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)

        # == flow ==
        labels_one_hot = torch.nn.functional.one_hot(labels, args.num_class).type(torch.cuda.FloatTensor)
        flow_labels = labels_one_hot.unsqueeze(1).cuda()
                    
                    
        loss_nll, log_p2 = flowTrainer.log_prob(flow_labels, feature_flow, flowNet)

        if args.lossType == "mix":
            if args.noise_mode=='asym': # Penalize confident prediction for asymmetric noise
                L = loss_ce + penalty + (args.lambda_f * loss_nll)
            else:
                L = loss_ce + (args.lambda_f * loss_nll)
        elif args.lossType == "ce":
            if args.noise_mode=='asym': # Penalize confident prediction for asymmetric noise
                L = loss_ce + penalty
            else:
                L = loss_ce
        elif args.lossType == "nll":
            if args.noise_mode=='asym':
                L = penalty + (args.lambda_f * loss_nll)
            else:
                L = (args.lambda_f * loss_nll)  

        L.backward()  
        optimizer.step()
        optimizer_flow.step()

        sys.stdout.write('\r')
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f nll loss: %.4f'
                %(batch_idx+1, args.num_batches, loss.item(), penalty.item(), loss_nll.item()))
        sys.stdout.flush()

        ## wandb
        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["loss/CEloss"] = loss.item()
            logMsg["loss/nll"] = loss_nll.item()
            logMsg["loss/penalty"] = penalty.item()
    
def eval(net, flowNet, loader):
    acc_meter.reset()
    net.eval()
    flowNet.eval()

    # flow acc
    correct_flow = 0
    prob_sum_flow = 0

    # cross entropy acc
    correct_ce = 0
    prob_sum_ce = 0

    correct_mix = 0
    prob_sum_mix = 0

    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            _, logits1, feature1 = net1(inputs, get_feature = True)
            outputs1 = flowTrainer.predict(flowNet1, feature1)

            _, logits2, feature2 = net2(inputs, get_feature = True)
            outputs2 = flowTrainer.predict(flowNet2, feature2)

            logits  = (torch.softmax(logits1, dim=1) + torch.softmax(logits2, dim=1)) / 2
            outputs = (outputs1 + outputs2) / 2
            mix_outs = (logits + outputs) / 2

            total   += targets.size(0)

            _ , predicted_net = torch.max(logits, 1)
            correct_ce += predicted_net.eq(targets).cpu().sum().item()

            _ , predicted_flow = torch.max(outputs, 1)
            correct_flow += predicted_flow.eq(targets).cpu().sum().item()    
                   
            _ , predicted_mix = torch.max(mix_outs, 1)
            correct_mix += predicted_mix.eq(targets).cpu().sum().item()

            if args.lossType == "ce":
                acc_meter.add(logits, targets)
            elif args.lossType == "nll":
                acc_meter.add(outputs, targets)
            elif args.lossType == "mix":
                acc_meter.add(mix_outs, targets)

    acc_ce = 100.*correct_ce/total
    acc_flow = 100.*correct_flow/total
    acc_mix = 100.*correct_mix/total

    if args.lossType == "ce":
        acc = acc_ce
        # confidence = confidence_ce
    elif args.lossType == "nll":
        acc = acc_flow
        # confidence = confidence_flow
    elif args.lossType == "mix":
        acc = acc_mix
        # confidence = confidence_mix

    print("\n| Test Acc: %.2f%%\n" %(acc))  
    accs = acc_meter.value()
    
    ## wandb
    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["accHead/test_flow"] = acc_flow
        logMsg["accHead/test_resnet"] = acc_ce
        logMsg["accHead/test_mix"] = acc_mix
        wandb.log(logMsg)

    return acc , accs

## Calculate the KL Divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen_Shannon divergence (Symmetric and Smoother than the KL divergence) 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

def Selection_Rate(args, prob):
    threshold = torch.mean(prob)
    if threshold.item() > args.d_u:
            threshold = threshold - (threshold-torch.min(prob))/args.tau
    if threshold.item() < args.d_up:
            threshold = threshold + (torch.max(prob) - threshold)/args.tau
    SR = torch.sum(prob<threshold).item()/args.num_samples
    print("threshold : ", torch.mean(prob))
    print("threshold(new) : ", threshold)
    print("prob size : ", prob.size())
    if SR <= 0.1  or SR >= 1.0:
        new_SR = np.clip(SR, 0.1 , 0.9)
        print(f'WARNING: sample rate = {SR}, set to {new_SR}')
        SR = new_SR
    return SR, threshold

## Calculate JSD
def Calculate_JSD(epoch, net1, flowNet1, ne2, flowNet2):
    net1.eval()
    net2.eval()
    flowNet1.eval()
    flowNet2.eval()

    num_samples = args.num_batches*args.batch_size
    prob = torch.zeros(num_samples)
    JS_dist = Jensen_Shannon()
    paths = []
    n=0
    eval_loader = loader.run(0, 'eval_train')
    for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda() 
        batch_size      = inputs.size()[0]
        ## Get the output of the Model
        with torch.no_grad():
            _, logits, feature = net1(inputs, get_feature = True)
            out1 = flowTrainer.predict(flowNet1, feature)

            _, logits2, feature2 = net2(inputs, get_feature = True)
            out2 = flowTrainer.predict(flowNet2, feature2)

        ## Get the Prediction
        if args.lossType == "ce":
            out = (torch.softmax(logits, dim=1) + torch.softmax(logits2, dim=1)) / 2
        elif args.lossType == "nll":
            out = (out1 + out2) / 2
        elif args.lossType == "mix":
            out = (torch.softmax(logits, dim=1) + torch.softmax(logits2, dim=1) + out1 + out2) / 4

       
        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        # dist = JS_dist(out,  F.one_hot(targets, num_classes = args.num_class))
        dist = js_distance(out, targets, args.num_class)
        prob[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist 
        
        for b in range(inputs.size(0)):
            paths.append(path[b])
            n+=1

        sys.stdout.write('\r')
        sys.stdout.write('| Evaluating loss Iter %3d\t' %(batch_idx)) 
        sys.stdout.flush()
            
    return prob,paths  

## Penalty for Asymmetric Noise    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

## Get the pre-trained model                
def get_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, args.num_class)
    return model 

def create_model():
    model = resnet50(num_classes=args.num_class, feature_dim=args.cond_size)
    model = model.cuda()
    return model

def logJSD_RealDataset(epoch, threshold, labeled_trainloader, unlabeled_trainloader):
    labeled_idx = labeled_trainloader.dataset.pred_idx
    unlabeled_idx = unlabeled_trainloader.dataset.pred_idx
    origin_prob =  labeled_trainloader.dataset.origin_prob
    labeled_prob = [origin_prob[i] for i in labeled_idx]
    unlabeled_prob = [origin_prob[i] for i in unlabeled_idx]
    sample_ratio = torch.sum(torch.from_numpy(origin_prob)<threshold).item()/origin_prob.shape[0]

    # draw JSD dis
    plt.clf()
    kwargs = dict(histtype='stepfilled', alpha=0.75, density=False, bins=20)
    plt.hist(origin_prob, color='blue', range=(0., 1.), label='prob', **kwargs)

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

        wandb.log(logMsg)

def load_model():
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1)))
    flowNet1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_flow_1)))
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_2)))
    flowNet2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_flow_2)))

def save_model(idx, net, flowNet):
    if idx == 0:
        save_point = os.path.join(model_save_loc, model_name_1)
        save_point_flow = os.path.join(model_save_loc, model_name_flow_1)
        torch.save(net.state_dict(), save_point)
        torch.save(flowNet.state_dict(), save_point_flow)
    else:
        save_point = os.path.join(model_save_loc, model_name_2)
        save_point_flow = os.path.join(model_save_loc, model_name_flow_2)
        torch.save(net.state_dict(), save_point)
        torch.save(flowNet.state_dict(), save_point_flow)

def run(idx, net1, flowNet1, net2, flowNet2, optimizer, optimizerFlow, nb_repeat):
    ## Calculate JSD values and Filter Rate
    print(f"Calculate JSD Net {idx}\n")
    prob, paths = Calculate_JSD(epoch, net1, flowNet1, net2, flowNet2)
    threshold   = torch.mean(prob)                                           ## Simply Take the average as the threshold
    SR = torch.sum(prob<threshold).item()/prob.size()[0]                    ## Calculate the Ratio of clean samples      

    for i in range(nb_repeat):
        print(f'Train Net {idx} repeat {i}\n')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= prob,  paths=paths) # Uniform Selection
        train(epoch, net1, flowNet1, net2, flowNet2, optimizer, optimizerFlow, labeled_trainloader, unlabeled_trainloader)    # train net1  
        acc_val, accs_val = eval(net1, flowNet1, val_loader)
        if acc_val > best_acc[idx-1]:
            print('| Saving Best Net%d ...'%idx)
            best_acc[idx-1] = acc_val
            save_model(idx, net1, flowNet1)

    logJSD_RealDataset(epoch, threshold, labeled_trainloader, unlabeled_trainloader)

## Semi-Supervised Loss
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up, lambda_u, linear_u):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch, warm_up, lambda_u, 0, linear_u)


log = open('./checkpoint/%s.txt'%args.id,'w')     
log.flush()

loader = dataloader.clothing_dataloader(root=args.data_path, batch_size=args.batch_size, warmup_batch_size = args.batch_size*4, num_workers=8, num_batches=args.num_batches)
print('| Building Net')

# flow model
flowTrainer = FlowTrainer(args)
flowNet1 = flowTrainer.create_model()
flowNet2 = flowTrainer.create_model()

model = get_model()
net1  = create_model()
net2  = create_model()
cudnn.benchmark = True

## Optimizer and Learning Rate Scheduler 
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizerFlow1 = optim.SGD(flowNet1.parameters(), lr=args.lr_f, momentum=0.9, weight_decay=1e-3)
optimizerFlow2 = optim.SGD(flowNet2.parameters(), lr=args.lr_f, momentum=0.9, weight_decay=1e-3)

scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 100, 1e-5)
schedulerFlow1 = optim.lr_scheduler.CosineAnnealingLR(optimizerFlow1, 100, args.lr_f / 100)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, 100, 1e-5)
schedulerFlow2 = optim.lr_scheduler.CosineAnnealingLR(optimizerFlow2, 100, args.lr_f / 100)

## Cross-Entropy and Other Losses
CE     = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()
criterion    = SemiLoss()

## Warm-up Epochs (maximum value is 2, we recommend 0 or 1)
warm_up = args.warm_up

## Copy Saved Data
if args.pretrained: 
    params_model  = model.named_parameters()
    params_net_1 = net1.named_parameters() 
    params_net_2 = net2.named_parameters() 

    dict_params_net_1 = dict(params_net_1)
    dict_params_net_2 = dict(params_net_2)

    for name, param in params_model:
        if name in dict_params_net_1:
            dict_params_net_1[name].data.copy_(param.data)
        if name in dict_params_net_2:
            dict_params_net_2[name].data.copy_(param.data)

## Location for saving the models 
folder = 'Clothing1M_flow' + '_' + str(args.name)
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)
    os.mkdir(model_save_loc + '/JSD_distribution')

## wandb
if (wandb != None):
    wandb.init(project="Clothing1M", entity="andy-su", name=folder)
    wandb.run.log_code(".")
    wandb.config.update(args)
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("acc/test", summary="max")

net1 = nn.DataParallel(net1)
net2 = nn.DataParallel(net2)
flowNet1 = nn.DataParallel(flowNet1)
flowNet2 = nn.DataParallel(flowNet2)

## Loading Saved Weights
model_name_1 = 'clothing1m_net_1.pth.tar'
model_name_flow_1 = 'clothing1m_flow_1.pth.tar'
model_name_2 = 'clothing1m_net_2.pth.tar'
model_name_flow_2 = 'clothing1m_flow_2.pth.tar'

if args.resume:
    load_model()

best_acc = [0,0]
SR = 0
torch.backends.cudnn.benchmark = True
acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
nb_repeat = 2

for epoch in range(0, args.num_epochs+1):
    startTime = time.time() 
    val_loader = loader.run(0, 'val')
    
    if epoch>100:
        nb_repeat =3  ## Change how many times we want to repeat on the same selection

    if epoch<warm_up:             
        print('Warmup Net 1')
        train_loader = loader.run(0,'warmup')
        warmup(net1, flowNet1, optimizer1, optimizerFlow1,train_loader)

        print('Warmup Net 2')
        train_loader = loader.run(0,'warmup')
        warmup(net2, flowNet2, optimizer2, optimizerFlow2,train_loader)

    else:
        run(0, net1, flowNet1, net2, flowNet2, optimizer1, optimizerFlow1, nb_repeat)
        run(1, net2, flowNet2, net1, flowNet1, optimizer2, optimizerFlow2, nb_repeat)

    scheduler1.step()
    scheduler2.step()
    schedulerFlow1.step()
    schedulerFlow2.step()

    # Validation
    acc_val1, accs_val1  = eval(net1, flowNet1, val_loader)
    acc_val2, accs_val2 = eval(net2, flowNet2, val_loader) 
    log.write('Validation Epoch:%d  Acc1:%.2f  Acc2:%.2f\n'%(epoch, acc_val1, acc_val2))
    
    load_model()

    log.flush() 
    test_loader = loader.run(0,'test')  
    acc_test, accs_test = eval(net, flowNet, test_loader)   
    print('\n| Epoch:%d \t  Acc: %.2f%% (%.2f%%) \n'%(epoch,accs_test[0],accs_test[1]))
    log.write('Epoch:%d \t  Acc: %.2f%% (%.2f%%) \n'%(epoch,accs_test[0],accs_test[1]))
    log.flush()  

    ## wandb
    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["runtime"] = time.time() - startTime
        logMsg["acc/test"] = acc_test
        logMsg["acc/test_top1"] = accs_test[0]
        logMsg["acc/test_top5"] = accs_test[1]
        
        logMsg["acc/val"] = acc_val1
        logMsg["acc/test_top1"] = accs_val1[0]
        logMsg["acc/test_top5"] = accs_val1[1]
        wandb.log(logMsg)
