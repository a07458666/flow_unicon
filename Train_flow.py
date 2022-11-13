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
    if args.useUncertainty:
        origin_densitys = labeled_trainloader.dataset.origin_densitys
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
    clean_density = []
    noise_density = []
    for idx_noise_zip in [zip(labeled_idx, cleanset_noise_mask), zip(unlabeled_idx, noiseset_noise_mask)]:
        for idx, is_noise in idx_noise_zip:
            p = origin_prob[idx]
            if args.useUncertainty:
                d = origin_densitys[idx]
            if is_noise == 1.0:
                noise_prob.append(float(p))
                if args.useUncertainty:
                    noise_density.append(np.abs(float(d)))
            else:
                clean_prob.append(float(p))
                if args.useUncertainty:
                    clean_density.append(np.abs(float(d)))

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
    
    if args.useUncertainty:
        plt.clf()
        kwargs = dict(histtype='stepfilled', alpha=0.75, density=False, bins=20)
        plt.hist(clean_density, color='green', label='clean', **kwargs)
        plt.hist(noise_density, color='red'  , label='noisy', **kwargs)

        plt.axvline(x=origin_densitys.mean(), color='gray')
        plt.xlabel('Density Values')
        plt.ylabel('count')
        plt.title(f'Density Distribution of N Samples epoch :{epoch}')
        # plt.xlim(-150., -50.)
        plt.grid(True)
        plt.savefig(f'{model_save_loc}/Density_distribution/epoch{epoch}.png')

        
        plt.clf()
        max_point = 200.
        vis_count = max_point * (len(clean_prob) / (len(clean_prob) + len(noise_prob)))
        for idx, (jsd_value, density_value) in tqdm(enumerate(zip(clean_prob, clean_density))):
            plt.scatter(jsd_value, density_value, c = 'green', s = 20, alpha = .5,marker = ".")
            if idx > vis_count:
                break

        for idx, (jsd_value, density_value) in tqdm(enumerate(zip(noise_prob, noise_density))):
                plt.scatter(jsd_value, density_value, c = 'blue', s = 20, alpha = .5,marker = ".")
                if idx > (max_point - vis_count):
                    break

        plt.xlabel('JSD')
        plt.ylabel('Uncertainty')
        plt.title(f' Joint distribution of loss values and epistemic uncertainty epoch :{epoch}')
        plt.grid(True)
        plt.savefig(f'{model_save_loc}/Density_distribution/epoch{epoch}_jsd_uncertainty.png')

    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg["JSD"] = wandb.Image(f'{model_save_loc}/JSD_distribution/epoch{epoch}.png')
        if args.useUncertainty:
            logMsg["Density"] = wandb.Image(f'{model_save_loc}/Density_distribution/epoch{epoch}.png')
            logMsg["Density"] = wandb.Image(f'{model_save_loc}/Density_distribution/epoch{epoch}_jsd_uncertainty.png')
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
    sample_ratio = torch.sum(prob<threshold).item()/args.num_samples

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

def load_model(model_save_loc, net, flowNet, flowTrainer, best=False):
    if best:
        model_name = 'Net.pth'
        model_name_flow = 'FlowNet.pth'
        model_name_ema = 'Net_ema.pth'
        model_name_flow_ema = 'FlowNet_ema.pth'
    else:
        model_name = 'Net_warmup.pth'
        model_name_flow = 'FlowNet_warmup.pth'
        model_name_ema = 'Net_warmup_ema.pth'
        model_name_flow_ema = 'FlowNet_warmup_ema.pth'

    device = torch.device('cuda', torch.cuda.current_device())
    net.load_state_dict(torch.load(os.path.join(model_save_loc, model_name), map_location=device)['net'])
    flowNet.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_flow), map_location=device)['net'])
    
    flowTrainer.net_ema.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_ema), map_location=device)['net'])
    flowTrainer.flowNet_ema.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_flow_ema), map_location=device)['net'])
    return

def save_model(net, flowNet, epoch, acc = 0):
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

if __name__ == '__main__':
    ## Arguments to pass 
    args = argumentParse()
    print("args : ",vars(args))

    ## GPU Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    # torch.cuda.set_device(args.gpuid)
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
    if not os.path.exists(model_save_loc + '/Density_distribution'):
        os.mkdir(model_save_loc + '/Density_distribution')

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
    
    ## Call the dataloader
    if args.dataset== 'cifar10' or args.dataset== 'cifar100':
        loader = dataloader(args.dataset, r=args.ratio, noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=args.num_workers,\
            root_dir=model_save_loc, noise_file='%s/clean_%.4f_%s.npz'%(args.data_path,args.ratio, args.noise_mode))
    elif args.dataset== 'TinyImageNet':
        loader = dataloader(root=args.data_path, batch_size=args.batch_size, num_workers=args.num_workers, ratio = args.ratio, noise_mode = args.noise_mode, noise_file='%s/clean_%.2f_%s.npz'%(args.data_path,args.ratio, args.noise_mode))
    elif args.dataset == 'WebVision':
        loader = dataloader(batch_size=args.batch_size,num_workers=args.num_workers,root_dir=args.data_path, num_class=args.num_class)

    print('| Building net')
    net = create_model(args)

    # flow model
    flowTrainer = FlowTrainer(args)
    flowNet = flowTrainer.create_model()

    # gpus
    if len(args.gpuid) > 1:
        # print("mutle gpu!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        net = nn.DataParallel(net)
        flowNet = nn.DataParallel(flowNet)

    cudnn.benchmark = True

    ## Optimizer and Scheduler
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
    optimizerFlow = optim.SGD(flowNet.parameters(), lr=args.lr_f, momentum=0.9, weight_decay=5e-4)
    # optimizerFlow = optim.AdamW(flowNet.parameters(), lr=args.lr_f)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, args.lr / 100)
    schedulerFlow = optim.lr_scheduler.CosineAnnealingLR(optimizerFlow, args.num_epochs, args.lr_f / 100)

    flowTrainer.setEma(net, flowNet)

    ## Resume from the warmup checkpoint 
    if args.resume:
        start_epoch = args.warm_up
        load_model(model_save_loc, net, flowNet, flowTrainer)
        
    elif args.pretrain != '':
        start_epoch = 0
        args.warm_up = 1
        net.load_state_dict(torch.load(args.pretrain)['net'])
    else:
        start_epoch = 0

    best_acc = 0
    # jsd_threshold = args.thr

    ## Warmup and SSL-Training 
    for epoch in range(start_epoch,args.num_epochs+1):
        startTime = time.time() 
        test_loader = loader.run(0, 'val')
        eval_loader = loader.run(0, 'eval_train')
        warmup_trainloader = loader.run(0,'warmup')
            
        print("Data Size : ", len(warmup_trainloader.dataset))
        ## Warmup Stage 
        if epoch<args.warm_up:       
            warmup_trainloader = loader.run(0, 'warmup')

            print('Warmup Model')
            flowTrainer.warmup_standard(epoch, net, flowNet, optimizer, optimizerFlow, warmup_trainloader)   
        
        else:
            ## Calculate JSD values and Filter Rate
            print("Calculate JSD")
            prob = flowTrainer.Calculate_JSD(net, flowNet, args.num_samples, eval_loader)
            SR , threshold = Selection_Rate(args, prob)
            print('Train Net\n')
            if args.useUncertainty:
                print("Calculate Density")
                density = flowTrainer.Calculate_Density(net, flowNet, args.num_samples, eval_loader)
                labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= prob, densitys=density) # Uniform Selection
            else:
                labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= prob) # Uniform Selection
            if args.isRealTask:
                logJSD_RealDataset(epoch, threshold, labeled_trainloader, unlabeled_trainloader)
            else:
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

            save_model(net, flowNet, epoch, acc)
            best_acc = acc