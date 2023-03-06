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
from torch.utils.data import Dataset, DataLoader

# 
import time
import collections.abc
from flow_trainer import FlowTrainer
from tqdm import tqdm
from config import argumentParse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# try:
#     import wandb
# except ImportError:
#     wandb = None
#     logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")
wandb = None

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
    # optimizer.load_state_dict(net_pth['optimizer'])
    # scheduler.load_state_dict(net_pth['scheduler'])

    model_epoch = net_pth['epoch']
    return model_epoch


# def save_model(path, net, optimizer, scheduler, acc):
#     if len(args.gpuid) > 1:
#         net_state_dict = net.module.state_dict()
#     else:
#         net_state_dict = net.state_dict()

#     checkpoint = {
#         'net': net_state_dict,
#         'optimizer': optimizer.state_dict(),
#         'scheduler': scheduler.state_dict(),
#         'Noise_Ratio': args.ratio,
#         'Optimizer': 'SGD',
#         'Noise_mode': args.noise_mode,
#         'Accuracy': acc,
#         'Dataset': args.dataset,
#         'Batch Size': args.batch_size,
#         'epoch': epoch,
#     }
#     torch.save(checkpoint, path)
#     return 

def drawLabelHist(name, labels):
    # langs=['0','1','2','3','4','5','6','7','8','9']
    plt.ylim(0, 1)
    plt.xlim(-0.7, 9.7)
    x = np.arange(args.num_class)
    width = 1 / (args.num_class + 1)
    for idx, label in enumerate(labels):
        offset = (width * idx) - 0.5
        plt.bar(x + offset, label, width=width)
    
    axes = plt.gca()
    axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    
    plt.title(f"{args.noise_mode}_{args.ratio}_{args.lossType}_{name}")
    plt.show()  
    plt.savefig(f"./vis/label_distribution_{args.noise_mode}_{args.ratio}_{args.lossType}_{name}.png")

def drawLabelHistOne(name, labels):
    x = np.arange(args.num_class)
    for idx, label in enumerate(labels):
        plt.clf()
        plt.ylim(0, 1)
        plt.xlim(-0.7, 9.7)
        plt.bar(x, label)
        axes = plt.gca()
        axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
        axes.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    
        plt.title(f"{args.noise_mode}_{args.ratio}_{args.lossType}_{name}_Idx_{idx}")
        plt.show()  
        plt.savefig(f"./vis/probability_{args.noise_mode}_{args.ratio}_{args.lossType}_{name}_Idx_{idx}.png")

def vis_distribution(net1, flowNet1, net2, flowNet2, loader):
    net1.eval()
    flowNet1.eval()
    net2.eval()
    flowNet2.eval()
    
    np.set_printoptions(precision=2)
    classes_labels_vec = []
    classes_labels_correct_vec = []
    
    
    class_ind = {}
    # clean label
    for kk in range(args.num_class):
        class_ind[kk] = [i for i,x in enumerate(loader.dataset.origin_label) if x==kk] 
    
    # noise label
    # class_ind = loader.dataset.class_ind
    
    for class_num in range(args.num_class):
        labels_vec = []
        labels_correct_vec = []
        print(f"class : {class_num } num : {len(class_ind[class_num])}")


        classes_idx = loader.dataset.class_ind[class_num]
        
        sampler = torch.utils.data.sampler.SubsetRandomSampler(classes_idx)

        sampler_loader = DataLoader(
                dataset=loader.dataset, 
                batch_size=500,
                sampler=sampler,
                num_workers=16,
                drop_last= True)
        
        # for idx in classes_idx:
        #     input = loader.dataset.train_data[idx].cuda()
        #     print(loader.dataset.train_data[idx].shape)
        #     print("classes : ", loader.dataset.noise_label[idx])
        with torch.no_grad():
            for batch_idx, (inputs, labels, blur) in enumerate(sampler_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                # print(labels[:10])
                _, outputs_1, features_1 = net1(inputs, get_feature = True)
                _, outputs_2, features_2 = net2(inputs, get_feature = True)
                
                flow_outputs_1 = flowTrainer.predict(flowNet1, features_1)
                flow_outputs_2 = flowTrainer.predict(flowNet2, features_2)
                
                if args.lossType == "mix":
                    out = ((torch.softmax(outputs_1, dim=1) + torch.softmax(outputs_2, dim=1) + flow_outputs_1 + flow_outputs_2) / 4)
                elif args.lossType == "nll":
                    out = ((flow_outputs_1 + flow_outputs_2) / 2)
                elif args.lossType == "ce":
                    out = ((torch.softmax(outputs_1, dim=1) + torch.softmax(outputs_2, dim=1)) / 2)
                
                labels_vec.append(out)
                prob, predicted = torch.max(out, 1)
                labels_correct_vec.append(out[predicted == class_num])
                
                # print("net1 : ", flow_outputs_1[:10].cpu().numpy())
                # print("net2 : ", flow_outputs_2[:10].cpu().numpy())
                # print("net_mix : ", ((flow_outputs_1 + flow_outputs_2) / 2)[:10].cpu().numpy())
                # print("out : ", out[:10].cpu().numpy())
                # print("out correct: ", out[predicted == class_num][:10].cpu().numpy())
                
        labels_vec = torch.cat(labels_vec, dim=0)
        labels_var = torch.var(input = labels_vec, dim=0)
        print("labels_var", labels_var)
        labels_dis = labels_vec.sum(dim=0, keepdim=False) / labels_vec.size(0)
        print("labels_dis : ", labels_dis)
        classes_labels_vec.append(labels_dis.cpu().numpy())
        
        labels_correct_vec = torch.cat(labels_correct_vec, dim=0)
        # labels_correct_var = torch.var(input = labels_correct_vec, dim=0)
        # print("labels_var", labels_var)
        labels_correct_dis = labels_correct_vec.sum(dim=0, keepdim=False) / labels_correct_vec.size(0)
        print("label_correct_dis : ", labels_correct_dis)
        classes_labels_correct_vec.append(labels_correct_dis.cpu().numpy())
        
    drawLabelHist("all", classes_labels_vec)
    drawLabelHist("correct", classes_labels_correct_vec)
    drawLabelHistOne("all", classes_labels_vec)
    return


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

    epoch = 0
    ## Resume from the warmup checkpoint 
    if args.resume:
        _ = load_model(os.path.join(model_save_loc, "Net_warmup_1.pth"), net1, optimizer1, scheduler1)
        _ = load_model(os.path.join(model_save_loc, "Net_warmup_2.pth"), net2, optimizer2, scheduler2)
        _ = load_model(os.path.join(model_save_loc, "FlowNet_warmup_1.pth"), flowNet1, optimizerFlow1, schedulerFlow1)
        epoch = load_model(os.path.join(model_save_loc, "FlowNet_warmup_2.pth"), flowNet2, optimizerFlow2, schedulerFlow2)

    elif args.resume_best:
        _ = load_model(os.path.join(model_save_loc, "Net_1.pth"), net1, optimizer1, scheduler1)
        _ = load_model(os.path.join(model_save_loc, "Net_2.pth"), net2, optimizer2, scheduler2)
        _ = load_model(os.path.join(model_save_loc, "FlowNet_1.pth"), flowNet1, optimizerFlow1, schedulerFlow1)
        epoch = load_model(os.path.join(model_save_loc, "FlowNet_2.pth"), flowNet2, optimizerFlow2, schedulerFlow2)
    
    
    eval_loader = loader.run(0, 'eval_train')
    vis_distribution(net1, flowNet1, net2, flowNet2, eval_loader)
    
    # prob = flowTrainer.Calculate_JSD(net1, flowNet1, net2, flowNet2, args.num_samples, eval_loader)
    
    # SR , threshold = Selection_Rate(args, prob)
    
    # labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= prob) # Uniform Selection