from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os
from autoaugment import CIFAR10Policy, ImageNetPolicy
from torchnet.meter import AUCMeter
import torch.nn.functional as F 
from Asymmetric_Noise import *
from sklearn.metrics import confusion_matrix



## If you want to use the weights and biases 
# import wandb
# wandb.init(project="noisy-label-project", entity="....")


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset): 
    def __init__(self, dataset, sample_ratio, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], densitys=[]): 
        
        self.r = r # noise ratio
        self.sample_ratio = sample_ratio
        self.transform = transform
        self.mode = mode
        # self.blur_label = []
        root_dir_save = root_dir

        if dataset == 'cifar10':
            root_dir = './data/cifar10/cifar-10-batches-py'        
            num_class =10         
        elif dataset=='cifar100':
            root_dir = './data/cifar100/cifar-100-python'
            num_class =100
        else:
            print(f"dataset not define {dataset}")

        ## For Asymmetric Noise (CIFAR10)    
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} 

        num_sample     = 50000
        self.class_ind = {}

        if self.mode=='val':
            if dataset=='cifar10':    
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            
            # for i in range(50000):
            #     self.blur_label.append(torch.abs(torch.normal(mean=0, std=torch.tensor([0.1] * num_class))))
            
            if os.path.exists(noise_file):             
                noise_label = np.load(noise_file)['label']
                noise_idx = np.load(noise_file)['index']
                idx       = list(range(50000))
                clean_idx = [x for x in idx if x not in noise_idx]
                for kk in range(num_class):
                    self.class_ind[kk] = [i for i,x in enumerate(noise_label) if x==kk]

            else:       ## Inject Noise   
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r*50000)            
                noise_idx = idx[:num_noise]
                
                if noise_mode == 'asym':
                    if dataset== 'cifar100':
                        noise_label, prob11 =  noisify_cifar100_asymmetric(train_label, self.r)
                    else:
                        for i in range(50000):
                            if i in noise_idx:
                                    noiselabel = self.transition[train_label[i]]
                                    noise_label.append(noiselabel)
                            else:
                                noise_label.append(train_label[i])   
                else:
                    for i in range(50000):
                        if i in noise_idx:
                            if noise_mode=='sym':
                                if dataset=='cifar10': 
                                    noiselabel = random.randint(0,9)
                                elif dataset=='cifar100':    
                                    noiselabel = random.randint(0,99)
                                noise_label.append(noiselabel)

                            elif noise_mode=='pair_flip':  
                                noiselabel = self.pair_flipping[train_label[i]]
                                noise_label.append(noiselabel)   
                    
                        else:
                            noise_label.append(train_label[i])   

                print("Save noisy labels to %s ..."%noise_file)        
                np.savez(noise_file, label = noise_label, index = noise_idx)          
                for kk in range(num_class):
                    self.class_ind[kk] = [i for i,x in enumerate(noise_label) if x==kk]    

            if self.mode == 'all' or self.mode == 'ssl':
                self.train_data = train_data
                self.noise_label = noise_label
                self.origin_label = train_label   # original label
            else:
                save_file = 'Clean_index_'+ str(dataset) + '_' +str(noise_mode) +'_' + str(self.r) + '.npz'
                save_file = os.path.join(root_dir_save, save_file)

                if self.mode == "labeled":
                    pred_idx  = np.zeros(int(self.sample_ratio*num_sample))
                    class_len = int(self.sample_ratio*num_sample/num_class)
                    size_pred = 0

                    ## Ranking-based Selection and Introducing Class Balance
                    for i in range(num_class):            
                        class_indices = self.class_ind[i]
                        prob1  = np.argsort(probability[class_indices].cpu().numpy())
                        size1 = len(class_indices)

                        try:
                            pred_idx[size_pred:size_pred+class_len] = np.array(class_indices)[prob1[0:class_len].astype(int)].squeeze()
                            size_pred += class_len
                        except:                            
                            pred_idx[size_pred:size_pred+size1] = np.array(class_indices)
                            size_pred += size1
                    
                    pred_idx = [int(x) for x in list(pred_idx)]
                    np.savez(save_file, index = pred_idx)

                    ## Weights for label refinement
                    self.origin_prob = torch.clone(probability)
                    if len(densitys) > 0:
                        self.origin_densitys = torch.clone(densitys)
                    probability[probability<0.5] = 0
                    self.probability = [1-probability[i] for i in pred_idx]

                elif self.mode == "unlabeled":
                    pred_idx = np.load(save_file)['index']
                    idx = list(range(num_sample))
                    pred_idx_noisy = [x for x in idx if x not in pred_idx]        
                    pred_idx = pred_idx_noisy   
                                    
                self.train_data = train_data[pred_idx]
                self.noise_label = np.array([noise_label[i] for i in pred_idx])
                self.origin_label = np.array([train_label[i] for i in pred_idx])    # original label
                self.pred_idx = pred_idx                            

    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            # blur = self.blur_label[index]
            o_target = self.origin_label[index]
            image = Image.fromarray(img)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)

            return img1, img2, img3, img4,  target, prob, o_target#, blur

        elif self.mode=='unlabeled':
            img = self.train_data[index]
            o_target = self.origin_label[index]
            image = Image.fromarray(img)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            return img1, img2, img3, img4, o_target
        
        elif self.mode=='ssl':
            img, target = self.train_data[index], self.noise_label[index]
            image = Image.fromarray(img)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            return img1, img2, img3, img4, target

        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            # blur = self.blur_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target#, blur

        elif self.mode=='val':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='val':
            return len(self.train_data)
        else:
            return len(self.test_data)   
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        
        if self.dataset=='cifar10':
            transform_weak_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            transform_strong_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_10,
                "unlabeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
                "labeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
            }

            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])

        elif self.dataset=='cifar100':
            transform_weak_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            transform_strong_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_100,
                "unlabeled": [
                            transform_weak_100,
                            transform_weak_100,
                            transform_strong_100,
                            transform_strong_100
                        ],
                "labeled": [
                            transform_weak_100,
                            transform_weak_100,
                            transform_strong_100,
                            transform_strong_100
                        ],
            }        
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
                   
    def run(self, sample_ratio, mode, pred=[], prob=[], densitys=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["warmup"], mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["labeled"], mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob, densitys=densitys)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                # batch_size=self.batch_size,
                batch_size=int(self.batch_size * (2 * sample_ratio)),
                shuffle=True,
                num_workers=self.num_workers, drop_last=True)  

            unlabeled_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["unlabeled"], mode="unlabeled", noise_file=self.noise_file, pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                # batch_size= int(self.batch_size/(2*sample_ratio)),
                batch_size= int(self.batch_size * (2 * (1 - sample_ratio))),
                shuffle=True,
                num_workers=self.num_workers, drop_last =True)    

            return labeled_trainloader, unlabeled_trainloader                
        
        elif mode=='val':
            test_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='val')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=500,
                shuffle=False,
                num_workers=self.num_workers, drop_last= True)          
            return eval_loader   
        elif mode=='ssl_train':
            ssl_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["unlabeled"], mode="ssl")
            ssl_trainloader = DataLoader(
                dataset=ssl_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, drop_last=True)
            return ssl_trainloader
        elif mode=='val_noise':
            all_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["warmup"], mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=False,
                num_workers=0)             
            return trainloader       
            
