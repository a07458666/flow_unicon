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
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
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
parser.add_argument('--flow_modules', default="14-14-14-14", type=str)
parser.add_argument('--tol', default=1e-5, type=float, help='flow atol, rtol')
parser.add_argument('--cond_size', default=512, type=int)
parser.add_argument('--lambda_f', default=1., type=float, help='flow nll loss weight')
parser.add_argument('--pseudo_std', default=0.2, type=float)

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
parser.add_argument('--centering', default=False, type=bool, help='use centering')
parser.add_argument('--center_momentum', default=0.95, type=float, help='use centering')

# Flow w_u
parser.add_argument('--linear_u', default=16, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_flow_u', default=1, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_flow_u_warmup', default=0.1, type=float, help='weight for unsupervised loss start value')

parser.add_argument('--lambda_u', default=30, type=float, help='weight for unsupervised loss')
parser.add_argument('--isRealTask', default=True, type=bool, help='')

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

def eval(evalType, net, flowNet, loader):
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
            
            _, logits, feature = net(inputs, get_feature = True)
            outputs = flowTrainer.predict(flowNet, feature)


            logits  = torch.softmax(logits, dim=1)
            outputs = outputs
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

    print(f"\n| {evalType} Acc: {acc:.2f}\n")  
    accs = acc_meter.value()
    
    ## wandb
    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg[f"accHead/{evalType}_flow"] = acc_flow
        logMsg[f"accHead/{evalType}_resnet"] = acc_ce
        logMsg[f"accHead/{evalType}_mix"] = acc_mix
        wandb.log(logMsg)

    return acc , accs

## For Standard Training 
    def warmup_standard(epoch, net, flowNet, optimizer, optimizerFlow, dataloader, updateCenter = False):
        flowNet.train()
        net.train()
        
        num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
        
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda() 
            labels_one_hot = torch.nn.functional.one_hot(labels, args.num_class).type(torch.cuda.FloatTensor)

            _, outputs, feature_flow = net(inputs, get_feature = True)
            flow_labels = labels_one_hot.unsqueeze(1).cuda()
            logFeature(feature_flow)      
  
            # == flow ==
            loss_nll, log_p2 = flowTrainer.log_prob(flow_labels, feature_flow, flowNet)
            # == flow end ===

            loss_ce = CEloss(outputs, labels)
            penalty = conf_penalty(outputs)

            if args.lossType == "mix":
                L = loss_ce + (args.lambda_f * loss_nll)
            elif args.lossType == "ce":
                L = loss_ce
            elif args.lossType == "nll":
                L = (args.lambda_f * loss_nll)

            optimizer.zero_grad()
            optimizerFlow.zero_grad()
            L.backward()
            optimizer.step()
            optimizerFlow.step()  
            
            if args.centering and updateCenter:
                    _, _ = flowTrainer.get_pseudo_label(net, flowNet, inputs, inputs, std = args.pseudo_std, updateCnetering = True)

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
            sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t NLL-loss: %.4f'
                    %(args.dataset, epoch, args.num_epochs, batch_idx+1, num_iter, loss_nll.item()))
            
            sys.stdout.flush()

## Calculate JSD
def Calculate_JSD(epoch, net1, flowNet1, net2, flowNet2):
    net1.eval()
    net2.eval()
    flowNet1.eval()
    flowNet2.eval()

    num_samples = args.num_batches*args.batch_size
    prob = torch.zeros(num_samples)
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

def print_label_status(targets_x, targets_u):
    refine_labels_x = [0] * args.num_class
    pseudo_labels_u = [0] * args.num_class
    
    for i in targets_u.max(dim=1).indices:
        pseudo_labels_u[i.item()] += 1

    for i in targets_x.max(dim=1).indices:
        refine_labels_x[i.item()] += 1

    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["label_count/pseudo_labels_u"] =  max(pseudo_labels_u)
        logMsg["label_count/refine_labels_x"] =  max(refine_labels_x)
        wandb.log(logMsg)

def load_model(net1, flowNet1, net2, flowNet2):
    print("load_model")
    net1.module.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1)))
    flowNet1.module.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_flow_1)))
    net2.module.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_2)))
    flowNet2.module.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_flow_2)))

def save_model(idx, net, flowNet):
    if idx == 0:
        save_point = os.path.join(model_save_loc, model_name_1)
        save_point_flow = os.path.join(model_save_loc, model_name_flow_1)
    else:
        save_point = os.path.join(model_save_loc, model_name_2)
        save_point_flow = os.path.join(model_save_loc, model_name_flow_2)

    torch.save(net.module.state_dict(), save_point)
    torch.save(flowNet.module.state_dict(), save_point_flow)

def run(idx, net1, flowNet1, net2, flowNet2, optimizer, optimizerFlow, nb_repeat):
    ## Calculate JSD values and Filter Rate
    print(f"Calculate JSD Net {idx}\n")
    prob, paths = Calculate_JSD(epoch, net1, flowNet1, net2, flowNet2)

    threshold   = torch.mean(prob)                                           ## Simply Take the average as the threshold
    SR = torch.sum(prob<threshold).item()/prob.size()[0]                    ## Calculate the Ratio of clean samples      

    for i in range(nb_repeat):
        print(f'Train Net {idx} repeat {i}\n')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= prob,  paths=paths) # Uniform Selection
        flowTrainer.train(epoch, net1, flowNet1, net2, flowNet2, optimizer, optimizerFlow, labeled_trainloader, unlabeled_trainloader)    # train net1  
        acc_val, accs_val = eval(f"val_{idx + 1}",net1, flowNet1, val_loader)
        if acc_val > best_acc[idx]:
            print(f'| Saving Best Net{idx} acc_val{acc_val:.3f}, best_acc {best_acc[idx]:.3f}')
            best_acc[idx] = acc_val
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
    wandb.define_metric("test/acc", summary="max")
    wandb.define_metric("val/acc", summary="max")
    

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
    load_model(net1, flowNet1, net2, flowNet2)

best_acc = [0,0]
SR = 0
torch.backends.cudnn.benchmark = True
acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
nb_repeat = 2

for epoch in range(0, args.num_epochs+1):
    startTime = time.time() 
    val_loader = loader.run(0, 'val')
    
    if epoch>100:
        nb_repeat = 3  ## Change how many times we want to repeat on the same selection

    if epoch<warm_up:             
        print('\nWarmup Net 1')
        warmup_trainloader = loader.run(0,'warmup')
        val_loader = loader.run(0, 'val')
        flowTrainer.warmup_standard(epoch, net1, flowNet1, optimizer1, optimizerFlow1,warmup_trainloader)
        acc_val1, accs_val1 = eval("val_1", net1, flowNet1, val_loader)
        save_model(0, net1, flowNet1)
        # if acc_val1 > best_acc[0]:
        #     print('| Saving Best Net%d ...'%0)
        #     best_acc[0] = acc_val1
        # save_model(0, net1, flowNet1)
        
        print('\nWarmup Net 2')
        warmup_trainloader = loader.run(0,'warmup')
        val_loader = loader.run(0, 'val')
        flowTrainer.warmup_standard(epoch, net2, flowNet2, optimizer2, optimizerFlow2,warmup_trainloader)
        acc_val2, accs_val2 = eval("val_2", net2, flowNet2, val_loader)
        save_model(1, net2, flowNet2)
        
        # if acc_val2 > best_acc[1]:
        #     print('| Saving Best Net%d ...'%2)
        #     best_acc[1] = acc_val2
        #     save_model(1, net2, flowNet2)

    else:
        run(0, net1, flowNet1, net2, flowNet2, optimizer1, optimizerFlow1, nb_repeat)
        run(1, net2, flowNet2, net1, flowNet1, optimizer2, optimizerFlow2, nb_repeat)

    scheduler1.step()
    scheduler2.step()
    schedulerFlow1.step()
    schedulerFlow2.step()

    # Validation
    # acc_val1, accs_val1 = eval("val_1", net1, flowNet1, val_loader)
    # acc_val2, accs_val2 = eval("val_2", net2, flowNet2, val_loader) 
    # print('\n|Validation Epoch:%d  Acc1:%.2f  Acc2:%.2f\n'%(epoch, acc_val1, acc_val2))
    # log.write('Validation Epoch:%d  Acc1:%.2f  Acc2:%.2f\n'%(epoch, acc_val1, acc_val2))
    
    acc_val, confidence_val  = flowTrainer.testByFlow(epoch, net1, flowNet1, net2, flowNet2, val_loader)
    
    load_model(net1, flowNet1, net2, flowNet2)
 
    test_loader = loader.run(0,'test')
    acc, confidence  = flowTrainer.testByFlow(epoch, net1, flowNet1, net2, flowNet2, test_loader)

    print(f"\n| Epoch:{epoch} \t  Test Acc: {acc:.2f} \n")
    log.write(f'\n| Epoch:{epoch} \t  Test Acc: {acc:.2f} \n')
    log.flush()  

    ## wandb
    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["runtime"] = time.time() - startTime
        logMsg["test/acc"] = acc
        logMsg["test/confidence"] = confidence
        logMsg["val/acc"] = acc_val
        logMsg["val/confidence_val"] = confidence_val

        
        wandb.log(logMsg)
