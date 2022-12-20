from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from autoaugment import CIFAR10Policy, ImageNetPolicy
import random
import numpy as np
from PIL import Image
import torch
import os

class imagenet_dataset(Dataset):
    def __init__(self, sample_ratio, root_dir, transform, num_class):
        self.sample_ratio = sample_ratio
        self.root = root_dir+'imagenet/val/'
        self.transform = transform
        self.val_data = []
        for c in range(num_class):
            imgs = os.listdir(self.root+str(c))
            for img in imgs:
                self.val_data.append([c,os.path.join(self.root,str(c),img)])                
                
    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')   
        img = self.transform(image) 
        return img, target
    
    def __len__(self):
        return len(self.val_data)

class webvision_dataset(Dataset): 
    def __init__(self, sample_ratio, root_dir, transform, mode, num_class, pred=[], probability=[]): 
        self.root = root_dir
        self.transform = transform
        self.mode = mode  
        
        num_samples = 65944

        if self.mode=='val':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img]=target                             
        else:    
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    train_imgs.append(img)
                    self.train_labels[img]=target
            save_file = 'Clean_index_webvision.npz'
            save_file = os.path.join(self.root, save_file)   
            if self.mode == 'all':
                self.train_imgs = train_imgs
            else:                   
                # if self.mode == "labeled":
                #     pred_idx = pred.nonzero()[0]
                #     self.train_imgs = [train_imgs[i] for i in pred_idx]                
                #     self.probability = [probability[i] for i in pred_idx]            
                #     print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))                                 
                # elif self.mode == "unlabeled":
                #     pred_idx = (1-pred).nonzero()[0]                                               
                #     self.train_imgs = [train_imgs[i] for i in pred_idx]                           
                #     print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))   
                if self.mode == "labeled":
                    sorted_indices  = np.argsort(probability.cpu().numpy())
                    clean_count = int(sample_ratio*num_samples)
                    pred_idx = sorted_indices[:clean_count]
                    np.savez(save_file, index = pred_idx)
                    # refine probability
                    self.origin_prob = torch.clone(probability).cpu().numpy()
                    probability[probability<0.5] = 0
                    self.probability = [1-probability[i] for i in pred_idx]
                    self.train_imgs  = [train_imgs[i] for i in pred_idx]
                    self.pred_idx = pred_idx
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
                elif self.mode == "unlabeled":
                    pred_idx_load = np.load(save_file)['index']
                    idx = list(range(num_samples))
                    pred_idx = [x for x in idx if x not in pred_idx_load]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]
                    self.pred_idx = pred_idx
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))



                    
    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform[0](image) 
            img2 = self.transform[1](image)
            img3 = self.transform[2](image) 
            img4 = self.transform[3](image)
            return img1, img2, img3, img4, target, prob, -1            
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform[0](image) 
            img2 = self.transform[1](image) 
            img3 = self.transform[2](image) 
            img4 = self.transform[3](image)
            return img1, img2, img3, img4, -1
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image)
            return img, target        
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
           
    def __len__(self):
        if self.mode!='val':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    


class webvision_dataloader():  
    def __init__(self, batch_size, num_class, num_workers, root_dir):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir

        transform_weak_compose = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) 

        transform_strong_compose = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) 
        
        self.transforms_train = {
            "warmup": transform_weak_compose,
            "unlabeled": [
                        transform_weak_compose,
                        transform_weak_compose,
                        transform_strong_compose,
                        transform_strong_compose
                    ],
            "labeled": [
                        transform_weak_compose,
                        transform_weak_compose,
                        transform_strong_compose,
                        transform_strong_compose
                    ],
        }    
            
        self.transform_test = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])  
        self.transform_imagenet = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])         

    def run(self, sample_ratio, mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = webvision_dataset(sample_ratio = sample_ratio, root_dir=self.root_dir, transform=self.transforms_train["warmup"], mode="all", num_class=self.num_class)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2 * 4,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)                 
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = webvision_dataset(sample_ratio = sample_ratio, root_dir=self.root_dir, transform=self.transforms_train["labeled"], mode="labeled",num_class=self.num_class,pred=pred,probability=prob)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=int(self.batch_size * 2 * sample_ratio),
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True)        
            
            unlabeled_dataset = webvision_dataset(sample_ratio = sample_ratio, root_dir=self.root_dir, transform=self.transforms_train["unlabeled"], mode="unlabeled",num_class=self.num_class,pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size * 2 * (1- sample_ratio)),
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='val':
            val_dataset = webvision_dataset(sample_ratio = sample_ratio, root_dir=self.root_dir, transform=self.transform_test, mode='val', num_class=self.num_class)      
            val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=self.batch_size * 2 * 4,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return val_loader
        
        elif mode=='eval_train':
            eval_dataset = webvision_dataset(sample_ratio = sample_ratio, root_dir=self.root_dir, transform=self.transform_test, mode='all', num_class=self.num_class)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size * 2 * 4,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True)               
            return eval_loader     
        
        elif mode=='imagenet':
            imagenet_val = imagenet_dataset(sample_ratio = sample_ratio, root_dir=self.root_dir, transform=self.transform_imagenet, num_class=self.num_class)      
            imagenet_loader = DataLoader(
                dataset=imagenet_val, 
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return imagenet_loader     
