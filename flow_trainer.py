from urllib3 import Retry
import torch
import torch.nn.functional as F

from flowModule.flow import cnf
from flowModule.utils import standard_normal_logprob, linear_rampup, mix_match, normal_logprob
from flowModule.logger import logFeature
from flowModule.jensen_shannon import js_distance

import time
import sys
from tqdm import tqdm
from Contrastive_loss import *
from torch_ema import ExponentialMovingAverage

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up, lambda_u, linear_u):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch, warm_up, lambda_u, 0, linear_u)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

class FlowTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.cond_size = args.cond_size
        self.warm_up = args.warm_up
        self.contrastive_criterion = SupConLoss()
        
        ## CE Loss Functions
        self.CE       = nn.CrossEntropyLoss(reduction='none')
        self.CEloss   = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss(reduction= 'none')
        self.criterion  = SemiLoss()

        self.conf_penalty = NegEntropy()

        ## centering
        self.center_momentum = args.center_momentum

        ## DA
        self.da_weight_ema = (torch.ones(args.num_class) / args.num_class).cuda()

        ## log_file
        # self.log_file = open('./log.txt','w')

        return

    ## For Standard Training 
    def warmup_standard(self, epoch, net, flowNet, optimizer, optimizerFlow, dataloader, updateCenter = False):
        flowNet.train()
        net.train()
        
        num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
        
        # for batch_idx, (inputs, labels, blur) in enumerate(dataloader):
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            labels_one_hot = torch.nn.functional.one_hot(labels, self.args.num_class).type(torch.cuda.FloatTensor)
            # if self.args.blur:
            #     blur = blur.cuda()
            #     labels_one_hot += blur
            if self.args.warmup_mixup:
                inputs, labels_one_hot = mix_match(inputs, labels_one_hot, self.args.alpha)    

            _, outputs, feature_flow = net(inputs, get_feature = True)
            flow_labels = labels_one_hot.unsqueeze(1).cuda()
            logFeature(feature_flow)      
  
            # == flow ==
            loss_nll, log_p2 = self.log_prob(flow_labels, feature_flow, flowNet)
            # == flow end ===

            loss_ce = self.CEloss(outputs, labels)
            penalty = self.conf_penalty(outputs)

            if self.args.lossType == "mix":
                if self.args.noise_mode=='asym': # Penalize confident prediction for asymmetric noise
                    L = loss_ce + penalty + (self.args.lambda_f * loss_nll)
                else:
                    L = loss_ce + (self.args.lambda_f * loss_nll)
            elif self.args.lossType == "ce":
                if self.args.noise_mode=='asym': # Penalize confident prediction for asymmetric noise
                    L = loss_ce + penalty
                else:
                    L = loss_ce
            elif self.args.lossType == "nll":
                if self.args.noise_mode=='asym':
                    L = penalty + (self.args.lambda_f * loss_nll)
                else:
                    L = (self.args.lambda_f * loss_nll)

            optimizer.zero_grad()
            optimizerFlow.zero_grad()
            L.backward()
            optimizer.step()
            optimizerFlow.step()  
            
            if self.args.centering and updateCenter:
                    _, _ = self.get_pseudo_label(net, flowNet, inputs, inputs, std = self.args.pseudo_std, updateCnetering = True)

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
            if self.args.isRealTask:
                sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t NLL-loss: %.4f'
                        %(self.args.dataset, epoch, self.args.num_epochs, batch_idx+1, num_iter, loss_nll.item()))
            else:
                sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t NLL-loss: %.4f'
                        %(self.args.dataset, self.args.ratio, self.args.noise_mode, epoch, self.args.num_epochs, batch_idx+1, num_iter, loss_nll.item()))
            sys.stdout.flush()

    def warmup_ssl_mixup(self, epoch, net, flowNet, optimizer, optimizerFlow, dataloader, updateCenter = False):
        flowNet.train()
        net.train()
        
        num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
        
        for batch_idx, (inputs_w1, inputs_w2, inputs_s3, inputs_s4, labels) in enumerate(dataloader): 
            inputs_w1, inputs_w2, inputs_s3, inputs_s4, labels = inputs_w1.cuda(), inputs_w2.cuda(), inputs_s3.cuda(), inputs_s4.cuda(), labels.cuda()
            
            labels_one_hot = torch.nn.functional.one_hot(labels, self.args.num_class).type(torch.cuda.FloatTensor)

            inputs, labels_one_hot = mix_match(inputs_w1, labels_one_hot, self.args.alpha)    

            _, outputs, feature_flow = net(inputs, get_feature = True)
            flow_labels = labels_one_hot.unsqueeze(1).cuda()
            logFeature(feature_flow)      
  
            # == flow ==
            loss_nll, log_p2 = self.log_prob(flow_labels, feature_flow, flowNet)
            # == flow end ===

            loss_ce = self.CEloss(outputs, labels)
            penalty = self.conf_penalty(outputs)
            
            # == Unsupervised Contrastive Loss ===
            # inputs_s34 = torch.cat([inputs_s3, inputs_s4], dim=0)
            # f, _ = net(inputs_s34)
            # f1 = f[:inputs_s3.size(0)]
            # f2 = f[inputs_s3.size(0):]
            # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # loss_simCLR = self.contrastive_criterion(features)
            # =================
            f1, _ = net(inputs_s3)
            f2, _ = net(inputs_s4)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_simCLR = self.contrastive_criterion(features)
            # == Unsupervised Contrastive Loss End ===

            if self.args.lossType == "mix":
                L = loss_ce + (self.args.lambda_f * loss_nll)
            elif self.args.lossType == "ce":
                L = loss_ce
            elif self.args.lossType == "nll":
                L = (self.args.lambda_f * loss_nll)
            L += loss_simCLR
            optimizer.zero_grad()
            optimizerFlow.zero_grad()
            L.backward()
            optimizer.step()
            optimizerFlow.step()  
            
            if self.args.centering and updateCenter:
                    _, _ = self.get_pseudo_label(net, flowNet, inputs, inputs, std = self.args.pseudo_std, updateCnetering = True)

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
            if self.args.isRealTask:
                sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t NLL-loss: %.4f'
                        %(self.args.dataset, epoch, self.args.num_epochs, batch_idx+1, num_iter, loss_nll.item()))
            else:
                sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t NLL-loss: %.4f'
                        %(self.args.dataset, self.args.ratio, self.args.noise_mode, epoch, self.args.num_epochs, batch_idx+1, num_iter, loss_nll.item()))
            sys.stdout.flush()
              
    def train(self, epoch, net1, flowNet1, net2, flowNet2, optimizer, optimizerFlow, labeled_trainloader, unlabeled_trainloader):
        net1.train()
        flowNet1.train()
        net2.eval()
        flowNet2.eval()

        unlabeled_train_iter = iter(unlabeled_trainloader)
        num_iter = (len(labeled_trainloader.dataset)//labeled_trainloader.batch_size)+1 

        # for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x, labels_x_o, blur) in enumerate(labeled_trainloader): 
        for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x, labels_x_o) in enumerate(labeled_trainloader): 
            try:
                inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u_o = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u_o = next(unlabeled_train_iter)
            
            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            labels_x_num = labels_x.cuda()
            labels_x = torch.zeros(batch_size, self.args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
            w_x = w_x.view(-1,1).type(torch.FloatTensor) 

            inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()
            
            # if self.args.blur:
            #     blur = blur.cuda()
            #     labels_x += blur

            lamb_Tu = linear_rampup(epoch+batch_idx/num_iter, self.warm_up, self.args.lambda_p, self.args.Tf_warmup, self.args.Tf)

            with torch.no_grad():
                # Label co-guessing of unlabeled samples        
                pu_net_1, pu_flow_1 = self.get_pseudo_label(net1, flowNet1, inputs_u, inputs_u2, std = self.args.pseudo_std)
                pu_net_2, pu_flow_2 = self.get_pseudo_label(net2, flowNet2, inputs_u, inputs_u2, std = self.args.pseudo_std)
                
                pu_net_sp_1 = self.sharpening(pu_net_1, self.args.T)
                pu_net_sp_2 = self.sharpening(pu_net_2, self.args.T)
                
                if self.args.flow_sp:
                    pu_flow_1 = self.sharpening(pu_flow_1, lamb_Tu)
                    pu_flow_2 = self.sharpening(pu_flow_2, lamb_Tu)


                ## Pseudo-label
                if self.args.lossType == "ce":
                    targets_u = (pu_net_sp_1 + pu_net_sp_2) / 2
                elif self.args.lossType == "nll":
                    targets_u = (pu_flow_1 + pu_flow_2) / 2
                elif self.args.lossType == "mix":
                    targets_u = (pu_net_sp_1 + pu_net_sp_2 + pu_flow_1 + pu_flow_2) / 4

                targets_u = targets_u.detach()        

                ## Label refinement
                px_net_1, px_flow_1 = self.get_pseudo_label(net1, flowNet1, inputs_x, inputs_x2, std = self.args.pseudo_std)
                # px_net_2, px_flow_2 = self.get_pseudo_label(net2, flowNet2, inputs_x2, inputs_x2, std = self.args.pseudo_std)

                if self.args.lossType == "ce":
                    px = px_net_1
                elif self.args.lossType == "nll":
                    px = px_flow_1
                elif self.args.lossType == "mix":
                    px = (px_net_1 + px_flow_1) / 2

                px_mix = w_x*labels_x + (1-w_x)*px

                targets_x = self.sharpening(px_mix, self.args.T)        
                targets_x = targets_x.detach()

                ## updateCnetering
                if self.args.centering:
                    _, _ = self.get_pseudo_label(net1, flowNet1, inputs_u, inputs_u2, std = self.args.pseudo_std, updateCnetering = True)   

                if not self.args.isRealTask:
                    labels_x_o = labels_x_o.cuda()
                    labels_u_o = labels_u_o.cuda()
                    
                    self.print_label_status(targets_x, targets_u, labels_x_o, labels_u_o, epoch)

                    # Calculate label sources
                    u_sources_pseudo = js_distance(targets_u, labels_u_o, self.args.num_class)
                    x_sources_origin = js_distance(labels_x, labels_x_o, self.args.num_class)
                    x_sources_refine = js_distance(targets_x, labels_x_o, self.args.num_class)

            ## Unsupervised Contrastive Loss
            if self.args.clr_loss:
                f1, _ = net1(inputs_u3)
                f2, _ = net1(inputs_u4)

                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss_simCLR = self.contrastive_criterion(features)
            else:
                loss_simCLR = torch.tensor(0).cuda()

            all_inputs  = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            mixed_input, mixed_target = mix_match(all_inputs, all_targets, self.args.alpha)
                    
            _, logits, flow_feature = net1(mixed_input, get_feature = True) # add flow_feature
        
            logits_x = logits[:batch_size*2]
            logits_u = logits[batch_size*2:] 
            
            if self.args.lossType == "mix" or self.args.lossType == "ce":
                ## Combined Loss
                Lx, Lu, lamb = self.criterion(logits_x, mixed_target[:batch_size*2],
                                              logits_u, mixed_target[batch_size*2:],
                                              epoch+batch_idx/num_iter, self.warm_up,
                                              self.args.lambda_u, self.args.linear_u)
                
                ## Regularization
                prior = torch.ones(self.args.num_class)/self.args.num_class
                prior = prior.cuda()        
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior*torch.log(prior/pred_mean))
                
                loss_ce = Lx + lamb * Lu + penalty

            ## Flow loss
            _, log_p2 = self.log_prob(mixed_target.unsqueeze(1).cuda(), flow_feature, flowNet1)

            lamb_u = linear_rampup(epoch+batch_idx/num_iter, self.warm_up, self.args.linear_u, self.args.lambda_flow_u_warmup, self.args.lambda_flow_u)
            
            loss_nll_x = -log_p2[:batch_size*2]
            loss_nll_u = -log_p2[batch_size*2:]


            log_p2[batch_size*2:] *= lamb_u
            loss_nll = (-log_p2).mean()

            loss_flow = (self.args.lambda_f * loss_nll)

            ## Total Loss
            if self.args.lossType == "mix":
                loss = loss_flow + loss_ce + (self.args.lambda_c * loss_simCLR)
            elif self.args.lossType == "ce":
                loss = loss_ce + (self.args.lambda_c * loss_simCLR)
            elif self.args.lossType == "nll":
                loss = loss_flow + (self.args.lambda_c * loss_simCLR)

            # Compute gradient and Do SGD step
            optimizer.zero_grad()
            optimizerFlow.zero_grad()
            flow_feature.retain_grad() # show grad
            loss.backward()
            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(flowNet1.parameters(), 1e-10)
            optimizer.step()
            optimizerFlow.step()  

            ## wandb
            if (wandb != None):
                logMsg = {}
                logMsg["epoch"] = epoch
                logMsg["lamb_Tu"] = lamb_Tu
                
                logMsg["loss/nll_x"] = loss_nll_x.mean().item()
                logMsg["loss/nll_u"] = loss_nll_u.mean().item()

                logMsg["loss/nll_x_max"] = loss_nll_x.max()
                logMsg["loss/nll_x_min"] = loss_nll_x.min()
                logMsg["loss/nll_x_var"] = loss_nll_x.var()
                logMsg["loss/nll_u_max"] = loss_nll_u.max()
                logMsg["loss/nll_u_min"] = loss_nll_u.min()
                logMsg["loss/nll_u_var"] = loss_nll_u.var()

                logMsg["loss/simCLR"] = loss_simCLR.item()

                if self.args.lossType == "mix" or self.args.lossType == "nll":
                    logMsg["feature_grad/mean"] = flow_feature.grad.mean().item()
                    logMsg["feature_grad/max"] = flow_feature.grad.max().item()
                    logMsg["feature_grad/min"] = flow_feature.grad.min().item()
                
                if not self.args.isRealTask:           
                    logMsg["label_quality/unlabel_pseudo_JSD_mean"] = u_sources_pseudo.mean().item()
                    logMsg["label_quality/label_origin_JSD_mean"] = x_sources_origin.mean().item()
                    logMsg["label_quality/label_refine_JSD_mena"] = x_sources_refine.mean().item()

                if self.args.lossType == "mix" or self.args.lossType == "ce":
                    logMsg["loss/ce_x"] = Lx.item()
                    logMsg["loss/mse_u"] = Lu.item()
                    logMsg["loss/penalty"] = penalty.item()

                if self.args.centering:
                    if len(self.args.gpuid) > 1:
                        logMsg["centering(max)"] = flowNet1.module.center.max().item()
                        logMsg["centering(min)"] = flowNet1.module.center.min().item()
                        logMsg["centering(min)"] = flowNet1.module.center.min().item()
                    else:
                        logMsg["centering(max)"] = flowNet1.center.max().item()
                        logMsg["centering(min)"] = flowNet1.center.min().item()
                        logMsg["centering(min)"] = flowNet1.center.min().item()

                wandb.log(logMsg)
            
            sys.stdout.write('\r')
            if self.args.isRealTask:
                sys.stdout.write(f"{self.args.dataset} | Epoch [{epoch:3d}/{self.args.num_epochs:3d}] Iter[{batch_idx+1:3d}/{num_iter}]\t Contrastive Loss:{loss_simCLR.item():.4f} NLL(x) loss: {loss_nll_x.mean():.2f} NLL(u) loss: {loss_nll_u.mean().item():.2f}")
            else:
                sys.stdout.write(f"{self.args.dataset}: {self.args.ratio:.2f}-{self.args.noise_mode} | Epoch [{epoch:3d}/{self.args.num_epochs:3d}] Iter[{batch_idx+1:3d}/{num_iter}]\t Contrastive Loss:{loss_simCLR.item():.4f} NLL(x) loss: {loss_nll_x.mean():.2f} NLL(u) loss: {loss_nll_u.mean().item():.2f}")
            sys.stdout.flush()

    ## Calculate JSD
    def Calculate_JSD(self, net1, flowNet1, net2, flowNet2, num_samples, eval_loader):  
        net1.eval()
        net2.eval()
        flowNet1.eval()
        flowNet2.eval()
        JSD   = torch.zeros(num_samples)    
        for batch_idx, (inputs, targets) in tqdm(enumerate(eval_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = inputs.size()[0]

            ## Get outputs of both network
            with torch.no_grad():
                _, logits, feature = net1(inputs, get_feature = True)
                out1 = self.predict(flowNet1, feature)

                _, logits2, feature2 = net2(inputs, get_feature = True)
                out2 = self.predict(flowNet2, feature2)

            ## Get the Prediction
            if self.args.lossType == "ce":
                out = (torch.softmax(logits, dim=1) + torch.softmax(logits2, dim=1)) / 2
            elif self.args.lossType == "nll":
                out = (out1 + out2) / 2
            elif self.args.lossType == "mix":
                out = (torch.softmax(logits, dim=1) + torch.softmax(logits2, dim=1) + out1 + out2) / 4
            
            ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
            dist = js_distance(out, targets, self.args.num_class)
            JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist
        return JSD

    def create_model(self):
        model = cnf(self.args.num_class, self.args.flow_modules, self.cond_size, 1, tol = self.args.tol).cuda()
        model = model.cuda()
        return model

    def predict(self, flowNet, feature, mean = 0, std = 0, sample_n = 50, centering = False, normalize = True):
        with torch.no_grad():
            batch_size = feature.size()[0]
            input_z = torch.normal(mean = mean, std = std, size=(batch_size, sample_n, self.args.num_class)).cuda()
            approx21 = flowNet(input_z, feature, None, reverse=True)
            if len(self.args.gpuid) > 1:
                approx21_center = approx21 - flowNet.module.center
            else:
                approx21_center = approx21 - flowNet.center
            
            if centering:
                approx21_center += (1 / self.args.num_class)
                self.update_center(flowNet, approx21)

            probs = torch.mean(approx21_center, dim=1, keepdim=False)
            
            if normalize:
                probs = torch.tanh(probs)
                probs = torch.clamp(probs, min=0)
                probs = F.normalize(probs, dim=1, p=1)
            return probs

    def testSTD(self, epoch, net1, flowNet1, net2, flowNet2, test_loader, sample_std = 0.2):
        print("\n====TestSTD====\n")
        net1.eval()
        flowNet1.eval()
        net2.eval()
        flowNet2.eval()
        
        total = 0

        # flow acc
        correct_flow = 0
        prob_sum_flow = 0

        # cross entropy acc
        correct_ce = 0
        prob_sum_ce = 0

        correct_mix = 0
        prob_sum_mix = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
                inputs, targets = inputs.cuda(), targets.cuda()

                _, logits1, feature1 = net1(inputs, get_feature = True)
                outputs1 = self.predict(flowNet1, feature1, std = sample_std)

                _, logits2, feature2 = net2(inputs, get_feature = True)
                outputs2 = self.predict(flowNet2, feature2,  std = sample_std)

                total += targets.size(0)

                logits  = (torch.softmax(logits1, dim=1) + torch.softmax(logits2, dim=1)) / 2
                outputs = (outputs1 + outputs2) / 2

                prob_ce, predicted_ce = torch.max(logits, 1)

                prob_flow, predicted_flow = torch.max(outputs, 1)

                prob_mix, predicted_mix = torch.max(0.5 * logits + 0.5 * outputs, 1)

                correct_ce += predicted_ce.eq(targets).cpu().sum().item()  
                prob_sum_ce += prob_ce.cpu().sum().item()

                correct_flow += predicted_flow.eq(targets).cpu().sum().item()  
                prob_sum_flow += prob_flow.cpu().sum().item()

                correct_mix += predicted_mix.eq(targets).cpu().sum().item()  
                prob_sum_mix += prob_mix.cpu().sum().item()

        acc_ce = 100.*correct_ce/total
        confidence_ce = prob_sum_ce/total
                
        acc_flow = 100.*correct_flow/total
        confidence_flow = prob_sum_flow/total

        acc_mix = 100.*correct_mix/total
        confidence_mix = prob_sum_mix/total

        if self.args.lossType == "ce":
            acc = acc_ce
            confidence = confidence_ce
        elif self.args.lossType == "nll":
            acc = acc_flow
            confidence = confidence_flow
        elif self.args.lossType == "mix":
            acc = acc_mix
            confidence = confidence_mix
        
        print("\n| Test Epoch #%d\t STD:%f\t Accuracy: %.2f%%\t Condifence: %.2f%%\n" %(epoch, sample_std, acc, confidence))
        
        ## wandb
        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["flow_distribution/Acc(STD:" + str(sample_std) +  ")"] = acc
            logMsg["flow_distribution/Confidence(STD:" + str(sample_std) +  ")"] = confidence
            wandb.log(logMsg)
        return 

    def testByFlow(self, epoch, net1, flowNet1, net2, flowNet2, test_loader, test_num = -1):
        print("\n====Test====\n")
        net1.eval()
        flowNet1.eval()
        net2.eval()
        flowNet2.eval()
        
        total = 0

        # flow acc
        correct_flow = 0
        prob_sum_flow = 0

        # cross entropy acc
        correct_ce = 0
        prob_sum_ce = 0

        correct_mix = 0
        prob_sum_mix = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
                inputs, targets = inputs.cuda(), targets.cuda()

                _, logits1, feature1 = net1(inputs, get_feature = True)
                outputs1 = self.predict(flowNet1, feature1)

                _, logits2, feature2 = net2(inputs, get_feature = True)
                outputs2 = self.predict(flowNet2, feature2)

                total += targets.size(0)

                logits  = (torch.softmax(logits1, dim=1) + torch.softmax(logits2, dim=1)) / 2
                outputs = (outputs1 + outputs2) / 2

                prob_ce, predicted_ce = torch.max(logits, 1)

                prob_flow, predicted_flow = torch.max(outputs, 1)

                prob_mix, predicted_mix = torch.max(0.5 * logits + 0.5 * outputs, 1)

                correct_ce += predicted_ce.eq(targets).cpu().sum().item()  
                prob_sum_ce += prob_ce.cpu().sum().item()

                correct_flow += predicted_flow.eq(targets).cpu().sum().item()  
                prob_sum_flow += prob_flow.cpu().sum().item()

                correct_mix += predicted_mix.eq(targets).cpu().sum().item()  
                prob_sum_mix += prob_mix.cpu().sum().item()

                if test_num > 0 and total >= test_num:
                    break

        acc_ce = 100.*correct_ce/total
        confidence_ce = prob_sum_ce/total
                
        acc_flow = 100.*correct_flow/total
        confidence_flow = prob_sum_flow/total

        acc_mix = 100.*correct_mix/total
        confidence_mix = prob_sum_mix/total

        if self.args.lossType == "ce":
            acc = acc_ce
            confidence = confidence_ce
        elif self.args.lossType == "nll":
            acc = acc_flow
            confidence = confidence_flow
        elif self.args.lossType == "mix":
            acc = acc_mix
            confidence = confidence_mix
        
        print("\n| Test Epoch #%d\t Accuracy: %.2f%%\t Condifence: %.2f%%\n" %(epoch, acc, confidence))
        
        ## wandb
        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["accHead/test_flow"] = acc_flow
            logMsg["accHead/test_resnet"] = acc_ce
            logMsg["accHead/test_mix"] = acc_mix
            wandb.log(logMsg)

        return acc, confidence

    def torch_onehot(self, y, Nclass):
        if y.is_cuda:
            y = y.type(torch.cuda.LongTensor)
        else:
            y = y.type(torch.LongTensor)
        y_onehot = torch.zeros((y.shape[0], Nclass)).type(y.type())
        # In your for loop
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        return y_onehot

    ## Pseudo-label
    def get_pseudo_label(self, net, flowNet, inputs_u, inputs_u2, std = 0, updateCnetering = False):
        _, outputs_u11, features_u11 = net(inputs_u, get_feature = True)
        _, outputs_u12, features_u12 = net(inputs_u2, get_feature = True)
        
        flow_outputs_u11 = self.predict(flowNet, features_u11, std = std, centering = (self.args.centering and updateCnetering))
        flow_outputs_u12 = self.predict(flowNet, features_u12, std = std, centering = False)
        
        pu_net = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2
        pu_flow = (flow_outputs_u11 + flow_outputs_u12) / 2
        return pu_net, pu_flow

    def setEma(self, net, flowNet):
        self.net_ema = ExponentialMovingAverage(net.parameters(), decay=self.args.decay)
        self.flowNet_ema = ExponentialMovingAverage(flowNet.parameters(), decay=self.args.decay)
        return
    
    def print_label_status(self, targets_x, targets_u, labels_x_o, labels_u_o, epoch):
        refine_labels_x = [0] * self.args.num_class
        target_labels_x = [0] * self.args.num_class

        pseudo_labels_u = [0] * self.args.num_class
        target_labels_u = [0] * self.args.num_class
        for i in targets_u.max(dim=1).indices:
            pseudo_labels_u[i.item()] += 1
        for i in labels_u_o:
            target_labels_u[i.item()] += 1

        for i in targets_x.max(dim=1).indices:
            refine_labels_x[i.item()] += 1
        for i in labels_x_o:
            target_labels_x[i.item()] += 1
        # label_count.write('\nepoch : ' + str(epoch))
        # label_count.write('\npseudo_labels_u : ' + str(pseudo_labels_u))
        # label_count.write('\ntarget_labels_u : ' + str(target_labels_u))
        # label_count.write('\nrefine_labels_x : ' + str(refine_labels_x))
        # label_count.write('\ntarget_labels_x : ' + str(target_labels_x))
        # label_count.flush()

        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["label_count/pseudo_labels_u"] =  max(pseudo_labels_u)
            logMsg["label_count/target_labels_u"] =  max(target_labels_u)
            logMsg["label_count/refine_labels_x"] =  max(refine_labels_x)
            logMsg["label_count/target_labels_x"] =  max(target_labels_x)
            wandb.log(logMsg)

    def log_pu(self, pu_flow, pu_net, gt, epoch):
        prob_flow, predicted_flow = torch.max(pu_flow, 1)
        prob_net, predicted_net = torch.max(pu_net, 1)

        total = gt.size(0)
        correct_flow = predicted_flow.eq(gt).cpu().sum().item()  
        prob_sum_flow = prob_flow.cpu().sum().item()

        correct_net = predicted_net.eq(gt).cpu().sum().item()  
        prob_sum_net = prob_net.cpu().sum().item()

        acc_flow = 100.*correct_flow/total
        confidence_flow = prob_sum_flow/total

        acc_net = 100.*correct_net/total
        confidence_net = prob_sum_net/total

        if (wandb != None):
            logMsg = {}
            logMsg["epoch"] = epoch
            logMsg["pseudo/acc_flow"] = acc_flow
            logMsg["pseudo/confidence_flow"] = confidence_flow
            logMsg["pseudo/acc_net"] = acc_net
            logMsg["pseudo/confidence_net"] = confidence_net
            wandb.log(logMsg)
        return
    
    def sharpening(self, labels, T):
        labels = labels**(1/T)
        return labels / labels.sum(dim=1, keepdim=True)

    # def sharpening_DINO(self, labels, T):
    #     labels_sp = labels / T
    #     m = nn.Softmax(dim=0)
    #     return m(labels_sp)

    # def sharpening_UNICON(self, labels, T):
    #     labels = labels**(1/T)
    #     return labels / labels.sum(dim=1, keepdim=True)

    def entropy(self, p):
        return -p.mul((p + 1e-10).log2()).sum(dim=1)
        # b = F.softmax(x, dim=dim) * F.log_softmax(x, dim=dim)
        # e = b.sum(dim=1)
        # return e

    def log_prob(self, target, feature, flowNet):
        delta_p = torch.zeros(target.shape[0], target.shape[1], 1).cuda() 
        approx21, delta_log_p2 = flowNet(target, feature, delta_p)
        
        approx2 = standard_normal_logprob(approx21).view(target.size()[0], -1).sum(1, keepdim=True)
        # approx2 = normal_logprob(approx21, std = 2).view(target.size()[0], -1).sum(1, keepdim=True)
        
        delta_log_p2 = delta_log_p2.view(target.size()[0], target.shape[1], 1).sum(1)
        log_p2 = (approx2 - delta_log_p2)
        nll = -log_p2.mean()
        return nll, log_p2

    @torch.no_grad()
    def update_center(self, flowNet, teacher_output):
        """
        Update center used for teacher output.
        """
        # print("teacher_output", teacher_output[:10])
        # sample_n = teacher_output.size(1)
        # batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # batch_center = torch.sum(batch_center, dim=1, keepdim=False)
        # torch.distributed.all_reduce(batch_center)
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        # batch_center = batch_center / (len(teacher_output))
        # batch_center = batch_center / sample_n
        batch_center = teacher_output.mean(dim=0).mean(dim=0, keepdim=True)
        # ema update
        if len(self.args.gpuid) > 1:
            flowNet.module.center = flowNet.module.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        else:
            flowNet.center = flowNet.center * self.center_momentum + batch_center * (1 - self.center_momentum)
