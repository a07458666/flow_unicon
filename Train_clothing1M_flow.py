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

# flow
from flow_trainer import FlowTrainer
from flowModule.utils import standard_normal_logprob, linear_rampup, mix_match

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")


parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')   ## Set the learning rate to 0.005 for faster training at the beginning
parser.add_argument('--lr_f', default=2e-5, type=float, help='initial flow learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=8, type=int)
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
parser.add_argument('--flow_modules', default="256-256-256-256", type=str)
parser.add_argument('--ema_decay', default=0.9, type=float, help='ema decay')
parser.add_argument('--cond_size', default=512, type=int)
parser.add_argument('--lambda_f', default=0.1, type=float, help='flow nll loss weight')
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
def train(epoch, net, flowNet, optimizer, optimizer_flow, labeled_trainloader, unlabeled_trainloader):
    net.train()         # Fix one network and train the other    
    flowNet.train()       

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter_un  = (len(unlabeled_trainloader.dataset)//args.batch_size)+1
    num_iter_lab = (len(labeled_trainloader.dataset)//args.batch_size)+1

    num_iter = num_iter_lab
    
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
            targets_u, targets_x = get_pseudo_target(net, flowNet, w_x, labels_x, inputs_u3, inputs_u4, inputs_x3, inputs_x4)
            with net_ema.average_parameters():
                with flowNet_ema.average_parameters():
                    targets_u_ema, targets_x_ema = get_pseudo_target(net, flowNet, w_x, labels_x, inputs_u3, inputs_u4, inputs_x3, inputs_x4)
            
        targets_u = (targets_u + targets_u_ema) / 2
        targets_x = (targets_x + targets_x_ema) / 2

        ## Mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l,1-l)

        ## Unsupervised Contrastive Loss
        f1, _ = net(inputs_u)
        f2, _ = net(inputs_u2)
        features    = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss_simCLR = contrastive_criterion(features)

        all_inputs  = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a , input_b   = all_inputs , all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        ## Mixing inputs 
        mixed_input  = (l * input_a[: batch_size * 2] + (1 - l) * input_b[: batch_size * 2])
        mixed_target = (l * target_a[: batch_size * 2] + (1 - l) * target_b[: batch_size * 2])
                
        _, logits, flow_feature = net(mixed_input, get_feature = True)

        Lx = -torch.mean(
            torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1)
        )

        # Regularization feature var
        reg_f_var_loss = torch.clamp(1-torch.sqrt(flow_feature.var(dim=0) + 1e-10), min=0).mean()
            
        ## Flow
        flow_mixed_target = mixed_target.unsqueeze(1).cuda()
        delta_p = torch.zeros(flow_mixed_target.shape[0], flow_mixed_target.shape[1], 1).cuda()
        approx21, delta_log_p2 = flowNet(flow_mixed_target, flow_feature, delta_p)

        approx2 = standard_normal_logprob(approx21).view(mixed_target.size()[0], -1).sum(1, keepdim=True)
        delta_log_p2 = delta_log_p2.view(flow_mixed_target.size()[0], flow_mixed_target.shape[1], 1).sum(1)
        log_p2 = (approx2 - delta_log_p2)
        loss_flow = (-log_p2).mean() * args.lambda_f

        ## Regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        # loss = Lx  + args.lambda_c*loss_simCLR + penalty
        loss = Lx + args.lambda_c*loss_simCLR + penalty + reg_f_var_loss + loss_flow

        ## Compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_flow.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_flow.step()
        # EMA step
        net_ema.update()
        flowNet_ema.update()

        sys.stdout.write('\r')
        sys.stdout.write('%s: | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Contrative Loss:%.4f nll Loss:%.4f'
                %(args.dataset,  epoch, args.num_epochs, batch_idx+1, num_iter, Lx, loss_simCLR, loss_flow))
        sys.stdout.flush()

        ## wandb
        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["loss/Lx"] = Lx.item()
            logMsg["loss/loss_simCLR"] = loss_simCLR.item()
            logMsg["loss/penalty"] = penalty.item()
            logMsg["loss/reg_f_var_loss"] = reg_f_var_loss.item()
            logMsg["loss/loss_flow"] = loss_flow.item()


def warmup(net, flowNet, optimizer, optimizer_flow,dataloader):
    net.train()
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
        delta_p = torch.zeros(flow_labels.shape[0], flow_labels.shape[1], 1).cuda()
        approx21, delta_log_p2 = flowNet(flow_labels, feature_flow, delta_p)
        
        approx2 = standard_normal_logprob(approx21).view(flow_labels.size()[0], -1).sum(1, keepdim=True)
        delta_log_p2 = delta_log_p2.view(flow_labels.size()[0], flow_labels.shape[1], 1).sum(1)
        log_p2 = (approx2 - delta_log_p2)
        loss_nll = -log_p2.mean()

        L = loss + penalty + loss_nll    
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
    
def val(net, flowNet,val_loader):
    net.eval()
    flowNet.eval()
    correct_net = 0
    correct_flow = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, outputs_net, outputs_flow = predict(inputs, flowNet, getPre = True)
            with net_ema.average_parameters():
                with flowNet_ema.average_parameters():
                    outputs_ema, outputs_ema_net, outputs_ema_flow = predict(inputs, flowNet, getPre = True)
            
            outputs_net = (outputs_net + outputs_ema_net) / 2
            outputs_flow = (outputs_flow + outputs_ema_flow) / 2
            outputs = (outputs + outputs_ema) / 2

            total += targets.size(0)

            _ , predicted_net = torch.max(outputs_net, 1)
            correct_net += predicted_net.eq(targets).cpu().sum().item()

            _ , predicted_flow = torch.max(outputs_flow, 1)
            correct_flow += predicted_flow.eq(targets).cpu().sum().item()           
            _ , predicted = torch.max(outputs, 1)
            correct += predicted.eq(targets).cpu().sum().item()        
            
    acc_net = 100.*correct_net/total
    acc_flow = 100.*correct_flow/total
    acc = 100.*correct/total
    print("\n| Validation\t Net Acc: %.2f%%" %(acc))  
    if acc > best_acc[0]:
        best_acc[0] = acc
        print('| Saving Best Net ...')
        save_point = os.path.join(model_save_loc, 'clothing1m_net.pth.tar')
        save_point_flow = os.path.join(model_save_loc, 'clothing1m_flow.pth.tar')
        torch.save(net.state_dict(), save_point)
        torch.save(flowNet.state_dict(), save_point_flow)
    return acc, acc_net, acc_flow

def test(net, flowNet, test_loader):
    acc_meter.reset()
    net.eval()
    flowNet.eval()
    correct_net = 0
    correct_flow = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs, outputs_net, outputs_flow = predict(inputs, flowNet, getPre = True)
            with net_ema.average_parameters():
                with flowNet_ema.average_parameters():
                    outputs_ema, outputs_ema_net, outputs_ema_flow = predict(inputs, flowNet, getPre = True)

            outputs_net = (outputs_net + outputs_ema_net) / 2
            outputs_flow = (outputs_flow + outputs_ema_flow) / 2
            outputs = (outputs + outputs_ema) / 2

            total   += targets.size(0)

            _ , predicted_net = torch.max(outputs_net, 1)
            correct_net += predicted_net.eq(targets).cpu().sum().item()

            _ , predicted_flow = torch.max(outputs_flow, 1)
            correct_flow += predicted_flow.eq(targets).cpu().sum().item()           
            _ , predicted = torch.max(outputs, 1)
            correct += predicted.eq(targets).cpu().sum().item()           
   
            acc_meter.add(outputs,targets)
            
    acc_net = 100.*correct_net/total
    acc_flow = 100.*correct_flow/total
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))  
    accs = acc_meter.value()
    return acc , accs, acc_net, acc_flow

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

## Calculate JSD
def Calculate_JSD(epoch, net, flowNet):
    net.eval()
    flowNet.eval()
    num_samples = args.num_batches*args.batch_size
    prob = torch.zeros(num_samples)
    JS_dist = Jensen_Shannon()
    paths = []
    n=0
    for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda() 
        batch_size      = inputs.size()[0]
        ## Get the output of the Model
        with torch.no_grad():
            outputs = predict(inputs, flowNet)
            with net_ema.average_parameters():
                with flowNet_ema.average_parameters():
                    outputs_ema = predict(inputs, flowNet)
            outputs = (outputs + outputs_ema) / 2
       
        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        dist = JS_dist(outputs,  F.one_hot(targets, num_classes = args.num_class))
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

## Threshold Adjustment 
def linear_rampup(current, warm_up, rampup_length=5):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

## Predict
def predict(inputs, flowNet, getPre = False):
    _, logits, feature = net(inputs, get_feature=True)
    out_net = torch.softmax(logits, dim=1)
    out_flow = flowTrainer.predict(flowNet, feature)
    if getPre:
        return (out_net + out_flow) / 2, out_net, out_flow
    else:    
        return (out_net + out_flow) / 2

def get_pseudo_target(net, flowNet, w_x, labels_x, inputs_u3, inputs_u4, inputs_x3, inputs_x4):
    # Label co-guessing of unlabeled samples
    pu_net, pu_flow = flowTrainer.get_pseudo_label(net, flowNet, inputs_u3, inputs_u4)
    pu_net_sp = flowTrainer.sharpening(pu_net, args.T)  ## Temparature Sharpening
    ptu = (pu_net_sp + pu_flow) / 2
    
    targets_u = ptu / ptu.sum(dim=1, keepdim=True)   ## Normalize
    targets_u = targets_u.detach()     

    ## Label refinement of labeled samples
    px_net, px_flow = flowTrainer.get_pseudo_label(net, flowNet, inputs_x3, inputs_x4)
    px = (px_net + px_flow) / 2       
    px = w_x*labels_x + (1-w_x)*px        
    ptx = flowTrainer.sharpening(px, args.T)  ## Temparature Sharpening        
                
    targets_x = ptx / ptx.sum(dim=1, keepdim=True)  ## normalize           
    targets_x = targets_x.detach()

    return targets_u, targets_x

## Semi-Supervised Loss
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)


log = open('./checkpoint/%s.txt'%args.id,'w')     
log.flush()

loader = dataloader.clothing_dataloader(root=args.data_path, batch_size=args.batch_size, warmup_batch_size = args.batch_size*2, num_workers=8, num_batches=args.num_batches)
print('| Building Net')

# flow model
flowTrainer = FlowTrainer(args)
flowNet = flowTrainer.create_model()

model = get_model()
net  = create_model()
cudnn.benchmark = True

## Optimizer and Learning Rate Scheduler 
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer_flow = optim.SGD(flowNet.parameters(), lr=args.lr_f, momentum=0.9, weight_decay=1e-3)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, 1e-5)
scheduler_flow = optim.lr_scheduler.CosineAnnealingLR(optimizer_flow, 100, args.lr_f / 100)

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
    params_net = net.named_parameters() 

    dict_params_net = dict(params_net)

    for name, param in params_model:
        if name in dict_params_net:
            dict_params_net[name].data.copy_(param.data)

## Location for saving the models 
folder = 'Clothing1M_flow' + '_' + str(args.name)
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)

## wandb
if (wandb != None):
    wandb.init(project="Clothing1M", entity="andy-su", name=folder)
    wandb.config.update(args)
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("acc/test", summary="max")

net = nn.DataParallel(net)
flowNet = nn.DataParallel(flowNet)

# EMA
net_ema = ExponentialMovingAverage(net.parameters(), decay=args.ema_decay)
flowNet_ema = ExponentialMovingAverage(flowNet.parameters(), decay=args.ema_decay)

## Loading Saved Weights
model_name = 'clothing1m_net.pth.tar'
model_name_flow = 'clothing1m_flow.pth.tar'

if args.resume:
    net.load_state_dict(torch.load(os.path.join(model_save_loc, model_name)))
    flowNet.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_flow)))

best_acc = [0]
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
        train_loader = loader.run(0,'warmup')
        print('Warmup Net')
        warmup(net, flowNet, optimizer, optimizer_flow,train_loader)
        acc_val, acc_val_net, acc_val_flow = val(net,flowNet, val_loader)
    else:
        num_samples = args.num_batches*args.batch_size
        eval_loader = loader.run(0.5,'eval_train')  
        prob, paths = Calculate_JSD(epoch, net, flowNet)                          ## Calculate the JSD distances 
        threshold   = torch.mean(prob)                                           ## Simply Take the average as the threshold
        SR = torch.sum(prob<threshold).item()/prob.size()[0]                    ## Calculate the Ratio of clean samples      
        
        for i in range(nb_repeat):
            print('\n\nTrain Net')
            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob=prob,  paths=paths)         ## Uniform Selection
            train(epoch, net, flowNet, optimizer, optimizer_flow, labeled_trainloader, unlabeled_trainloader)                        ## Train Net
            acc_val, acc_val_net, acc_val_flow = val(net, flowNet,val_loader)

    scheduler.step()
    scheduler_flow.step()        
    # acc_val = val(net,val_loader)
    log.write('Validation Epoch:%d  Acc1:%.2f\n'%(epoch,acc_val))

    net.load_state_dict(torch.load(os.path.join(model_save_loc, 'clothing1m_net.pth.tar')))
    log.flush() 
    test_loader = loader.run(0,'test')  
    acc_test, accs, acc_test_net, acc_test_flow = test(net, flowNet, test_loader)   
    print('\n| Epoch:%d \t  Acc: %.2f%% (%.2f%%) \n'%(epoch,accs[0],accs[1]))
    log.write('Epoch:%d \t  Acc: %.2f%% (%.2f%%) \n'%(epoch,accs[0],accs[1]))
    log.flush()  

    ## wandb
    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["runtime"] = time.time() - startTime
        logMsg["acc/test"] = acc_test
        logMsg["acc/test_net"] = acc_test_net
        logMsg["acc/test_flow"] = acc_test_flow
        logMsg["acc/val"] = acc_val
        logMsg["acc/val_net"] = acc_val_net
        logMsg["acc/val_flow"] = acc_val_flow
        wandb.log(logMsg)
