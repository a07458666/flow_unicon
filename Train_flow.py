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

def logJSD(epoch, threshold, labeled_trainloader, unlabeled_trainloader):
    labeled_idx = labeled_trainloader.dataset.pred_idx
    unlabeled_idx = unlabeled_trainloader.dataset.pred_idx
    origin_prob =  labeled_trainloader.dataset.origin_prob

    labeled_prob = [origin_prob[i] for i in labeled_idx]
    unlabeled_prob = [origin_prob[i] for i in unlabeled_idx]
    sample_ratio = torch.sum(origin_prob < threshold).item()/args.num_samples

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
    clean_density = []
    noise_density = []
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

def logJSD_RealDataset(epoch, threshold, labeled_trainloader, unlabeled_trainloader):
    labeled_idx = labeled_trainloader.dataset.pred_idx
    unlabeled_idx = unlabeled_trainloader.dataset.pred_idx
    origin_prob =  labeled_trainloader.dataset.origin_prob
    labeled_prob = [origin_prob[i] for i in labeled_idx]
    unlabeled_prob = [origin_prob[i] for i in unlabeled_idx]
    sample_ratio = torch.sum(torch.from_numpy(origin_prob)<threshold).item()/args.num_samples

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

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model(args):
    if args.dataset=='WebVision':
        model = InceptionResNetV2(num_classes=args.num_class, feature_dim=args.cond_size)
    else:
        model = ResNet18(num_classes=args.num_class, feature_dim=args.cond_size)
    model = model.cuda()
    return model

def load_model(path, net, optimizer, scheduler):
    device = torch.device('cuda', torch.cuda.current_device())
    net_pth = torch.load(path, map_location=device)
    net.load_state_dict(net_pth['net'])
    optimizer.load_state_dict(net_pth['optimizer'])
    scheduler.load_state_dict(net_pth['scheduler'])

    model_epoch = net_pth['epoch']
    return model_epoch


def save_model(path, net, optimizer, scheduler, acc):
    if len(args.gpuid) > 1:
        net_state_dict = net.module.state_dict()
    else:
        net_state_dict = net.state_dict()

    checkpoint = {
        'net': net_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'Noise_Ratio': args.ratio,
        'Optimizer': 'SGD',
        'Noise_mode': args.noise_mode,
        'Accuracy': acc,
        'Dataset': args.dataset,
        'Batch Size': args.batch_size,
        'epoch': epoch,
    }
    torch.save(checkpoint, path)
    return 

def run(idx, net1, flowNet1, net2, flowNet2, optimizer, optimizerFlow):
    ## Calculate JSD values and Filter Rate
    print("Calculate JSD Net ", str(idx), "\n")
    prob = flowTrainer.Calculate_JSD(net1, flowNet1, net2, flowNet2, args.num_samples, eval_loader)
    SR , threshold = Selection_Rate(args, prob)
    
    print('Train Net ', str(idx), '\n')
    labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= prob) # Uniform Selection
    
    if args.isRealTask:
        logJSD_RealDataset(epoch, threshold, labeled_trainloader, unlabeled_trainloader)
    else:
        logJSD(epoch, threshold, labeled_trainloader, unlabeled_trainloader)

    flowTrainer.train(epoch, net1, flowNet1, net2, flowNet2, optimizer, optimizerFlow, labeled_trainloader, unlabeled_trainloader)    # train net1  

def manually_learning_rate(epoch, optimizer1, optimizerFlow1, optimizer2, optimizerFlow2, init_lr, init_flow_lr, mid_warmup = 25):
    lr = init_lr
    lr_flow = init_flow_lr
    if epoch >= 60 or (epoch+1)%mid_warmup==0:
        lr /= 10
        lr_flow /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr 
    for param_group in optimizerFlow1.param_groups:
        param_group['lr'] = lr_flow       
    for param_group in optimizerFlow2.param_groups:
        param_group['lr'] = lr_flow  

if __name__ == '__main__':
    ## Arguments to pass 
    args = argumentParse()
    print("args : ",vars(args))

    ## GPU Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
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
    elif args.dataset=='WebVision':
        from InceptionResNetV2 import *
        from dataloader_webvision import webvision_dataloader as dataloader

    ## Checkpoint Location
    if  args.isRealTask:
        folder = args.dataset +  '_flow_' + args.name
    else:
        folder = args.dataset + '_' + args.noise_mode + '_' + str(args.ratio)  + '_flow_' + args.name

    model_save_loc = './checkpoint/' + folder
    if not os.path.exists(model_save_loc):
        os.mkdir(model_save_loc)
    if not os.path.exists(model_save_loc + '/JSD_distribution'):
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
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            project_name = "FlowUNICON"
        else:
            project_name = "FlowUNICON_" + args.dataset
        wandb.init(project=project_name, entity="andy-su", name=folder)
        wandb.run.log_code(".")
        wandb.config.update(args)
        wandb.define_metric("acc/test", summary="max")
        wandb.define_metric("loss/nll", summary="min")
        wandb.define_metric("loss/nll_max", summary="min")
        wandb.define_metric("loss/nll_min", summary="min")
        wandb.define_metric("loss/nll_var", summary="min")
    
    ## Call the dataloader
    if args.dataset == 'cifar10' or args.dataset== 'cifar100':
        loader = dataloader(args.dataset, r=args.ratio, noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=args.num_workers,\
            root_dir=model_save_loc, noise_file='%s/clean_%.4f_%s.npz'%(args.data_path,args.ratio, args.noise_mode))
    elif args.dataset == 'TinyImageNet':
        loader = dataloader(root=args.data_path, batch_size=args.batch_size, num_workers=args.num_workers, ratio = args.ratio, noise_mode = args.noise_mode, noise_file='%s/clean_%.2f_%s.npz'%(args.data_path,args.ratio, args.noise_mode))
    elif args.dataset == 'WebVision':
        loader = dataloader(batch_size=args.batch_size,num_workers=args.num_workers,root_dir=args.data_path, num_class=args.num_class)

    print('| Building net')
    net1 = create_model(args)
    net2 = create_model(args)

    # flow model
    flowTrainer = FlowTrainer(args)
    flowNet1 = flowTrainer.create_model()
    flowNet2 = flowTrainer.create_model()

    cudnn.benchmark = True

    ## Optimizer and Scheduler
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay) 
    if args.optimizer == 'SGD':
        optimizerFlow1 = optim.SGD(flowNet1.parameters(), lr=args.lr_f, momentum=0.9, weight_decay=args.weight_decay)
        optimizerFlow2 = optim.SGD(flowNet2.parameters(), lr=args.lr_f, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizerFlow1 = optim.AdamW(flowNet1.parameters(), lr=args.lr_f)
        optimizerFlow2 = optim.AdamW(flowNet2.parameters(), lr=args.lr_f)

    if args.dataset=='TinyImageNet':
        scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer1, 0.98)
        schedulerFlow1 = optim.lr_scheduler.ExponentialLR(optimizerFlow1, 0.98)
        scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer2, 0.98)
        schedulerFlow2 = optim.lr_scheduler.ExponentialLR(optimizerFlow2, 0.98)
    else:
        scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, args.num_epochs, args.lr / 1e2)
        schedulerFlow1 = optim.lr_scheduler.CosineAnnealingLR(optimizerFlow1, args.num_epochs, args.lr_f / 1e2)
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, args.num_epochs, args.lr / 1e2)
        schedulerFlow2 = optim.lr_scheduler.CosineAnnealingLR(optimizerFlow2, args.num_epochs, args.lr_f / 1e2)

    ## Resume from the warmup checkpoint 
    if args.resume:
        _ = load_model(os.path.join(model_save_loc, "Net_warmup_1.pth"), net1, optimizer1, scheduler1)
        _ = load_model(os.path.join(model_save_loc, "Net_warmup_2.pth"), net2, optimizer2, scheduler2)
        _ = load_model(os.path.join(model_save_loc, "FlowNet_warmup_1.pth"), flowNet1, optimizerFlow1, schedulerFlow1)
        start_epoch = load_model(os.path.join(model_save_loc, "FlowNet_warmup_2.pth"), flowNet2, optimizerFlow2, schedulerFlow2)

    elif args.resume_best:
        _ = load_model(os.path.join(model_save_loc, "Net_1.pth"), net1, optimizer1, scheduler1)
        _ = load_model(os.path.join(model_save_loc, "Net_2.pth"), net2, optimizer2, scheduler2)
        _ = load_model(os.path.join(model_save_loc, "FlowNet_1.pth"), flowNet1, optimizerFlow1, schedulerFlow1)
        start_epoch = load_model(os.path.join(model_save_loc, "FlowNet_2.pth"), flowNet2, optimizerFlow2, schedulerFlow2)
        
    # elif args.pretrain != '':
    #     start_epoch = 0
    #     args.warm_up = 1
    #     net.load_state_dict(torch.load(args.pretrain)['net'])
    else:
        start_epoch = 0

    # gpus
    if len(args.gpuid) > 1:
        net1 = nn.DataParallel(net1)
        flowNet1 = nn.DataParallel(flowNet1)
        net2 = nn.DataParallel(net2)
        flowNet2 = nn.DataParallel(flowNet2)

    best_acc = 0

    if args.jumpRestart:
        mid_warmup = 25

    ## Warmup and SSL-Training 
    for epoch in range(start_epoch,args.num_epochs+1):
        startTime = time.time() 
        test_loader = loader.run(0, 'val')
        eval_loader = loader.run(0, 'eval_train')
        warmup_trainloader = loader.run(0,'warmup')

        if args.dataset == 'WebVision':
            imagenet_valloader = loader.run(0.5, 'imagenet')
        
        if args.dataset=='WebVision':
            manually_learning_rate(epoch, optimizer1, optimizerFlow1, optimizer2, optimizerFlow2, args.lr, args.lr_f, mid_warmup)
        print("Data Size : ", len(warmup_trainloader.dataset))
        
        ## Warmup Stage 
        if args.dataset=='WebVision':
            ssl_trainloader = loader.run(0.0, 'ssl')
            print('\nWarmup Model Net 1 (SSL & mixup)')
            flowTrainer.warmup_ssl_mixup(epoch, net1, flowNet1, optimizer1, optimizerFlow1, ssl_trainloader)
            print('\nWarmup Model Net 2 (SSL & mixup)')
            flowTrainer.warmup_ssl_mixup(epoch, net2, flowNet2, optimizer2, optimizerFlow2, ssl_trainloader)
            
        elif epoch<args.warm_up:       
            warmup_trainloader = loader.run(0, 'warmup')

            print('\nWarmup Model Net 1')
            flowTrainer.warmup_standard(epoch, net1, flowNet1, optimizer1, optimizerFlow1, warmup_trainloader)

            print('\nWarmup Model Net 2')
            flowTrainer.warmup_standard(epoch, net2, flowNet2, optimizer2, optimizerFlow2, warmup_trainloader)   
        ## Jump-Restart
        elif args.jumpRestart and (epoch+1) % mid_warmup == 0:
            manually_learning_rate(epoch, optimizer1, optimizerFlow1, optimizer2, optimizerFlow2, args.lr, args.lr_f, mid_warmup)

            warmup_trainloader = loader.run(0.5, 'warmup')
            print('Mid-training Warmup Net1')
            flowTrainer.warmup_standard(epoch, net1, flowNet1, optimizer1, optimizerFlow1, warmup_trainloader, updateCenter=True)   
            print('\nMid-training Warmup Net2')
            flowTrainer.warmup_standard(epoch, net2, flowNet2, optimizer2, optimizerFlow2, warmup_trainloader, updateCenter=True)   
        else:
            run(1, net1, flowNet1, net2, flowNet2, optimizer1, optimizerFlow1)
            run(2, net2, flowNet2, net1, flowNet1, optimizer2, optimizerFlow2)
        
        ## Acc
        acc, confidence = flowTrainer.testByFlow(epoch, net1, flowNet1, net2, flowNet2, test_loader)
        if args.dataset == 'WebVision':
            imagenet_acc, imagenet_confidence = flowTrainer.testByFlow(epoch, net1, flowNet1, net2, flowNet2, imagenet_valloader)
        if args.testSTD:
            for test_std in [0.0, 0.2, 0.5, 0.8, 1.0]:
                flowTrainer.testSTD(epoch, net1, flowNet1, net2, flowNet2, test_loader, sample_std = test_std)

        ## Acc(Train Dataset)
        # print('\n =====Noise Acc====')
        # noise_valloader = loader.run(0, 'val_noise')
        # acc_n, confidence_n = flowTrainer.testByFlow(epoch, net1, flowNet1, net2, flowNet2, noise_valloader, test_num = 5000)
        # print('\n ==================')
        
        if not (args.dataset=='WebVision'):
            scheduler1.step()
            schedulerFlow1.step()
            scheduler2.step()
            schedulerFlow2.step()

        ## wandb
        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["runtime"] = time.time() - startTime
            logMsg["acc/test"] = acc
            logMsg["confidence score"] = confidence
            if args.dataset == 'WebVision':
                logMsg["ImageNet/acc"] = imagenet_acc
                logMsg["ImageNet/confidence score"] = imagenet_confidence
            wandb.log(logMsg)
        
        if args.save_last:
            save_model(os.path.join(model_save_loc, "Net_last_1.pth"), net1, optimizer1, scheduler1, acc)
            save_model(os.path.join(model_save_loc, "Net_last_2.pth"), net2, optimizer2, scheduler2, acc)
            save_model(os.path.join(model_save_loc, "FlowNet_last_1.pth"), flowNet1, optimizerFlow1, schedulerFlow1, acc)
            save_model(os.path.join(model_save_loc, "FlowNet_last_2.pth"), flowNet2, optimizerFlow2, schedulerFlow2, acc)
        if acc > best_acc:
            if epoch <args.warm_up:
                save_model(os.path.join(model_save_loc, "Net_warmup_1.pth"), net1, optimizer1, scheduler1, acc)
                save_model(os.path.join(model_save_loc, "Net_warmup_2.pth"), net2, optimizer2, scheduler2, acc)
                save_model(os.path.join(model_save_loc, "FlowNet_warmup_1.pth"), flowNet1, optimizerFlow1, schedulerFlow1, acc)
                save_model(os.path.join(model_save_loc, "FlowNet_warmup_2.pth"), flowNet2, optimizerFlow2, schedulerFlow2, acc)
            else:
                save_model(os.path.join(model_save_loc, "Net_1.pth"), net1, optimizer1, scheduler1, acc)
                save_model(os.path.join(model_save_loc, "Net_2.pth"), net2, optimizer2, scheduler2, acc)
                save_model(os.path.join(model_save_loc, "FlowNet_1.pth"), flowNet1, optimizerFlow1, schedulerFlow1, acc)
                save_model(os.path.join(model_save_loc, "FlowNet_2.pth"), flowNet2, optimizerFlow2, schedulerFlow2, acc)
            best_acc = acc