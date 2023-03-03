from urllib3 import Retry
import torch
import torch.nn.functional as F

from flowModule.flow import cnf
from flowModule.utils import standard_normal_logprob, linear_rampup, mix_match
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
        self.mean = 0
        self.std = 0.2
        self.sample_n = 1
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
        return

    ## For Standard Training 
    def warmup_standard(self, epoch, net, flownet, optimizer, optimizerFlow, dataloader):
        flownet.train()
        if self.args.fix == 'net':
            net.eval()
        else:    
            net.train()
        num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda() 
            labels_one_hot = torch.nn.functional.one_hot(labels, self.args.num_class).type(torch.cuda.FloatTensor)
            _, outputs, feature_flow = net(inputs, get_feature = True)
            flow_labels = labels_one_hot.unsqueeze(1).cuda()
            logFeature(feature_flow)            

            # == flow ==
            loss_nll, log_p2 = self.log_prob(flow_labels, feature_flow, flownet)
            # == flow end ===

            if self.args.w_ce:
                loss_ce = self.CEloss(outputs, labels)
                if self.args.noise_mode=='asym':     # Penalize confident prediction for asymmetric noise
                    penalty = self.conf_penalty(outputs)
                    L = loss_ce + penalty + (self.args.lambda_f * loss_nll)
                else:   
                    L = loss_ce + (self.args.lambda_f * loss_nll)
            else:
                L = (self.args.lambda_f * loss_nll)

            optimizer.zero_grad()
            optimizerFlow.zero_grad()
            L.backward()
            if self.args.fix == 'flow':
                optimizer.step()
            elif self.args.fix == 'net':
                optimizerFlow.step()   
            else:
                optimizer.step()
                optimizerFlow.step()  
            
            self.net_ema.update()
            self.flowNet_ema.update()

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

    def train(self, epoch, net, flownet, optimizer, optimizerFlow, labeled_trainloader, unlabeled_trainloader):
        net.train()
        flownet.train()

        unlabeled_train_iter = iter(unlabeled_trainloader)
        num_iter = (len(labeled_trainloader.dataset)//labeled_trainloader.batch_size)+1 

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

            lamb_Tu = linear_rampup(epoch+batch_idx/num_iter, self.warm_up, self.args.lambda_p, self.args.Tu_warmup, self.args.Tu)

            with torch.no_grad():
                # Label co-guessing of unlabeled samples   
                with self.net_ema.average_parameters():
                    with self.flowNet_ema.average_parameters():
                        pu_net_ema, pu_flow_ema = self.get_pseudo_label(net, flownet, inputs_u, inputs_u2, std = self.args.pseudo_std, updateCnetering = True)        
                pu_net, pu_flow = self.get_pseudo_label(net, flownet, inputs_u, inputs_u2, std = self.args.pseudo_std)
                
                pu_net_sp = self.sharpening(pu_net, lamb_Tu)
                pu_net_ema_sp = self.sharpening(pu_net_ema, lamb_Tu)
                
                if self.args.flow_sp:
                    pu_flow_ema_sp = self.sharpening(pu_flow_ema, lamb_Tu)
                    pu_flow_sp = self.sharpening(pu_flow, lamb_Tu)
                else:
                    pu_flow_ema_sp = pu_flow_ema
                    pu_flow_sp = pu_flow

                pu_jsd_dist = js_distance(pu_flow, pu_flow_ema, self.args.num_class).mean()
                pu_entropy = self.entropy(pu_flow_ema).mean()

                ## Pseudo-label
                if self.args.pred == 'mixEMA':
                    if self.args.w_ce:
                        targets_u = (pu_flow_sp + pu_net_sp + pu_flow_ema_sp + pu_net_ema_sp) / 4
                    else:
                        targets_u = (pu_flow_ema_sp + pu_flow_sp) / 2
                elif self.args.pred == 'onlyEMA':
                    if self.args.w_ce:
                        targets_u = (pu_flow_ema_sp + pu_net_ema_sp) / 2
                    else:
                        targets_u = pu_flow_ema_sp

                targets_u = targets_u.detach()        

                ## Label refinement
                with self.net_ema.average_parameters():
                    with self.flowNet_ema.average_parameters():
                        px_net_ema, px_flow_ema = self.get_pseudo_label(net, flownet, inputs_x, inputs_x2, std = self.args.pseudo_std)

                px_net, px_flow = self.get_pseudo_label(net, flownet, inputs_x, inputs_x2, std = self.args.pseudo_std)
                px_jsd_dist = js_distance(px_flow, px_flow_ema, self.args.num_class).mean()
                px_entropy = self.entropy(px_flow_ema).mean()

                if self.args.pred == 'mixEMA':
                    if self.args.w_ce:
                        px = (px_flow + px_net + px_flow_ema + px_net_ema) / 4
                    else:
                        px = (px_flow_ema + px_flow) / 2
                elif self.args.pred == 'onlyEMA':
                    if self.args.w_ce:
                        px = (px_flow_ema + px_net_ema) / 2
                    else:
                        px = px_flow_ema

                px_mix = w_x*labels_x + (1-w_x)*px

                targets_x = self.sharpening(px_mix, lamb_Tu)        
                targets_x = targets_x.detach()

                if not self.args.isRealTask:
                    labels_x_o = labels_x_o.cuda()
                    labels_u_o = labels_u_o.cuda()

                    # self.log_pu((pu_flow_sp + pu_flow_ema_sp) / 2, (pu_net_sp + pu_net_ema_sp) / 2, labels_u_o, epoch)
                    self.log_pu(pu_flow_ema_sp, pu_net_ema_sp, labels_u_o, epoch)
                    
                    self.print_label_status(targets_x, targets_u, labels_x_o, labels_u_o, epoch)

                    # logFeature(torch.cat([features_u11_flow, features_u12_flow, features_x_flow, features_x2_flow], dim=0))

                    # Calculate label sources
                    u_sources_pseudo = js_distance(targets_u, labels_u_o, self.args.num_class)
                    x_sources_origin = js_distance(labels_x, labels_x_o, self.args.num_class)
                    x_sources_refine = js_distance(targets_x, labels_x_o, self.args.num_class)

            ## Supervised Contrastive Loss
            if self.args.supcon:
                fx_1, _ = net(inputs_x3)
                fx_2, _ = net(inputs_x4)
                features_x = torch.cat([fx_1.unsqueeze(1), fx_2.unsqueeze(1)], dim=1)
                loss_supCon = self.contrastive_criterion(features_x, labels_x_num)

            ## Unsupervised Contrastive Loss
            f1, _ = net(inputs_u3)
            f2, _ = net(inputs_u4)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_simCLR = self.contrastive_criterion(features)


            all_inputs  = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            mixed_input, mixed_target = mix_match(all_inputs, all_targets, self.args.alpha)
                    
            _, logits, flow_feature = net(mixed_input, get_feature = True) # add flow_feature

            # Regularization feature var
            reg_f_var_loss = torch.clamp(1-torch.sqrt(flow_feature.var(dim=0) + 1e-10), min=0).mean()
            
            logits_x = logits[:batch_size*2]
            logits_u = logits[batch_size*2:]        
            
            if self.args.w_ce:
                ## Combined Loss
                Lx, Lu, lamb = self.criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, self.warm_up, self.args.lambda_u, self.args.linear_u)
                
                ## Regularization
                prior = torch.ones(self.args.num_class)/self.args.num_class
                prior = prior.cuda()        
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior*torch.log(prior/pred_mean))
                loss_unicon = Lx + lamb * Lu + penalty

            ## Flow loss
            _, log_p2 = self.log_prob(mixed_target.unsqueeze(1).cuda(), flow_feature, flownet)

            lamb_u = linear_rampup(epoch+batch_idx/num_iter, self.warm_up, self.args.linear_u, self.args.lambda_flow_u_warmup, self.args.lambda_flow_u)
            
            loss_nll_x = -log_p2[:batch_size*2].mean()
            loss_nll_u = -log_p2[batch_size*2:].mean()
            if self.args.split:
                loss_flow = loss_nll_x + (lamb_u * loss_nll_u)
            else:
                loss_flow = (-log_p2).mean()

            ## Total Loss
            loss = self.args.lambda_c * loss_simCLR + (self.args.lambda_f * loss_flow) + reg_f_var_loss
            if self.args.w_ce:
                loss += loss_unicon
            if self.args.supcon:
                loss += self.args.lambda_c * loss_supCon

            # Compute gradient and Do SGD step
            optimizer.zero_grad()
            optimizerFlow.zero_grad()
            flow_feature.retain_grad() # show grad
            loss.backward()
            if self.args.fix == 'flow':
                optimizer.step()
            elif self.args.fix == 'net':
                optimizerFlow.step()  
            else:
                optimizer.step()
                optimizerFlow.step()  

            self.net_ema.update()
            self.flowNet_ema.update()

            ## wandb
            if (wandb != None):
                logMsg = {}
                logMsg["epoch"] = epoch
                logMsg["lamb_Tu"] = lamb_Tu
                
                logMsg["loss/nll_x"] = loss_nll_x.item()
                logMsg["loss/nll_u"] = loss_nll_u.item()

                logMsg["loss/nll_x_max"] = loss_nll_x.max()
                logMsg["loss/nll_x_min"] = loss_nll_x.min()
                logMsg["loss/nll_x_var"] = loss_nll_x.var()
                logMsg["loss/nll_u_max"] = loss_nll_u.max()
                logMsg["loss/nll_u_min"] = loss_nll_u.min()
                logMsg["loss/nll_u_var"] = loss_nll_u.var()

                logMsg["loss/simCLR"] = loss_simCLR.item()
                logMsg["loss/reg_f_var_loss"] = reg_f_var_loss.item()

                logMsg["collapse/px_jsd_dist"] = px_jsd_dist.item()
                logMsg["collapse/pu_jsd_dist"] = pu_jsd_dist.item()
                logMsg["entropy/px_entropy"] = px_entropy.item()
                logMsg["entropy/pu_entropy"] = pu_entropy.item()

                logMsg["feature_grad/mean"] = flow_feature.grad.mean().item()
                logMsg["feature_grad/max"] = flow_feature.grad.max().item()
                logMsg["feature_grad/min"] = flow_feature.grad.min().item()
                
                if not self.args.isRealTask:
                    logMsg["label_quality/unlabel_pseudo_JSD_mean"] = u_sources_pseudo.mean().item()
                    logMsg["label_quality/label_origin_JSD_mean"] = x_sources_origin.mean().item()
                    logMsg["label_quality/label_refine_JSD_mena"] = x_sources_refine.mean().item()

                if self.args.w_ce:
                    logMsg["loss/ce_x"] = Lx.item()
                    logMsg["loss/mse_u"] = Lu.item()
                    logMsg["loss/penalty"] = penalty.item()

                if self.args.centering:
                    logMsg["centering(max)"] = flownet.center.max().item()
                    logMsg["centering(min)"] = flownet.center.min().item()
                    logMsg["centering(min)"] = flownet.center.min().item()

                wandb.log(logMsg)

            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Contrastive Loss:%.4f NLL(x) loss: %.2f NLL(u) loss: %.2f'
                    %(self.args.dataset, self.args.ratio, self.args.noise_mode, epoch, self.args.num_epochs, batch_idx+1, num_iter, loss_simCLR.item(),  loss_nll_x.mean().item(), loss_nll_u.mean().item()))
            sys.stdout.flush()
        

    ## Calculate JSD
    def Calculate_JSD(self, net, flowNet, num_samples, eval_loader):  
        JSD   = torch.zeros(num_samples)    
        for batch_idx, (inputs, targets) in tqdm(enumerate(eval_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = inputs.size()[0]

            ## Get outputs of both network
            with torch.no_grad():
                _, _, feature = net(inputs, get_feature=True)
                out1 = self.predict(flowNet, feature)

                with self.net_ema.average_parameters():
                    with self.flowNet_ema .average_parameters():
                        _, _, feature2 = net(inputs, get_feature=True)
                        out2 = self.predict(flowNet, feature2)

            ## Get the Prediction
            out = (out1 + out2) / 2
            ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
            dist = js_distance(out, targets, self.args.num_class)
            JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist
        return JSD

    ## Calculate Density
    def Calculate_Density(self, net, flowNet, num_samples, eval_loader):  
        densitys   = torch.zeros(num_samples)  
        input_y = torch.zeros(size=(eval_loader.batch_size * self.args.num_class, 1, self.args.num_class)).cuda()
        
        # sample_n = self.args.num_class
        sample_n = 1

        for b in range(eval_loader.batch_size):
            for c in range(self.args.num_class):
                input_y[self.args.num_class * b + c, 0, c] = 1.

        for batch_idx, (inputs, targets) in tqdm(enumerate(eval_loader)):

            # to one hot
            targets = torch.zeros(eval_loader.batch_size, self.args.num_class).scatter_(1, targets.view(-1,1), 1).unsqueeze(1)

            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = inputs.size()[0]

            ## Get outputs of both network
            with torch.no_grad():
                _, _, feature = net(inputs, get_feature=True)
                feature = feature.repeat(sample_n, 1, 1)
                # print("input_y", input_y.size())
                # print("feature", feature.size())
                
                _, log_p2 = self.log_prob(targets, feature, flowNet)
                
                density1 = log_p2.view(sample_n, batch_size)
                
                density1_mean = torch.mean(density1, dim=0, keepdim=False)
            

                with self.net_ema.average_parameters():
                    with self.flowNet_ema .average_parameters():
                        _, _, feature = net(inputs, get_feature=True)
                        feature = feature.repeat(sample_n, 1, 1)
                        _, log_p2 = self.log_prob(targets, feature, flowNet)
                        density2 = log_p2.view(sample_n, batch_size)
                        density2_mean = torch.mean(density2, dim=0, keepdim=False)
                        

            ## Get the Prediction
            density = (density1_mean + density2_mean) / 2
            # print("d size :", density.size())
            densitys[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = density
        return densitys

    ## Calculate Uncertainty
    def Calculate_Uncertainty(self, net, flowNet, num_samples, eval_loader):  
        print("Calculate_Uncertainty Rand")
        uncertaintys   = torch.zeros(num_samples)  
        
        # sample_n = self.args.num_class
        sample_n = 100

        for batch_idx, (inputs, targets) in tqdm(enumerate(eval_loader)):
            
            # to one hot
            targets = torch.zeros(eval_loader.batch_size, self.args.num_class).scatter_(1, targets.view(-1,1), 1).unsqueeze(1)

            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = inputs.size()[0]

            ## Get outputs of both network
            mean = 0
            std = 1
            with torch.no_grad():
                with self.net_ema.average_parameters():
                    with self.flowNet_ema .average_parameters():
                        _, logits, feature = net(inputs, get_feature = True)
                        batch_size = feature.size()[0]
                        feature = feature.repeat(sample_n, 1, 1)
                        input_z = torch.normal(mean = mean, std = std, size=(sample_n * batch_size , self.args.num_class)).unsqueeze(1).cuda()
                        # input_z = torch.rand(sample_n * batch_size , self.args.num_class).unsqueeze(1).cuda()
                        delta_p = torch.zeros(input_z.shape[0], input_z.shape[1], 1).cuda()

                        approx21, _ = flowNet(input_z, feature, delta_p, reverse=True)
                        probs = torch.tanh(approx21)
                        # print("probs tanh", probs.size(), probs[:1])
                        probs = torch.clamp(probs, min=0, max=1)
                        # print("probs clamp", probs.size(), probs[:1])
                        probs = F.normalize(probs, dim=2, p=1)
                        # print("probs norm", probs.size(), probs[:1])
                        probs = probs.view(sample_n, batch_size, self.args.num_class)
                        probs_mean = torch.mean(probs, dim=0, keepdim=False)
                        # print("probs", probs.size())
                        entropy_val = self.entropy(probs_mean)
                        # print("entropy_val", entropy_val.size())
                        
                        
                        # print("probs_mean", probs_mean.size())

            ## Get the Prediction
            uncertaintys[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = entropy_val
        return uncertaintys

    def create_model(self):
        model = cnf(self.args.num_class, self.args.flow_modules, self.cond_size, 1).cuda()
        model = model.cuda()
        return model

    def predict(self, flowNet, feature, mean = 0, std = 0, sample_n = 4, centering = False):
        with torch.no_grad():
            batch_size = feature.size()[0]
            feature = feature.repeat(sample_n, 1, 1)
            input_z = torch.normal(mean = mean, std = std, size=(sample_n * batch_size , self.args.num_class)).unsqueeze(1).cuda()
            delta_p = torch.zeros(input_z.shape[0], input_z.shape[1], 1).cuda()

            approx21, _ = flowNet(input_z, feature, delta_p, reverse=True)
            approx21 = torch.clamp(approx21, min=0)
            approx21 = approx21 - flowNet.center

            probs = torch.tanh(approx21)
            probs = torch.clamp(probs, min=0, max=1)
            probs = probs.view(sample_n, -1, self.args.num_class)
            probs_mean = torch.mean(probs, dim=0, keepdim=False)
            probs_mean = F.normalize(probs_mean, dim=1, p=1)
            if centering:
                self.update_center(flowNet, approx21)
            return probs_mean

    def testByFlow(self, epoch, net, flownet, test_loader, test_num = -1):
        print("\n====Test====\n")
        net.eval()
        flownet.eval()
        total = 0
        # flow acc
        correct_flow = 0
        prob_sum_flow = 0

        # cross entropy acc
        correct_ce = 0
        prob_sum_ce = 0

        # mix acc
        correct_mix = 0
        prob_sum_mix = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
                inputs, targets = inputs.cuda(), targets.cuda()

                _, logits, feature = net(inputs, get_feature = True)
                outputs = self.predict(flownet, feature)

                with self.net_ema.average_parameters():
                    with self.flowNet_ema.average_parameters():
                        _, logits_ema, feature_ema = net(inputs, get_feature = True)
                        outputs_ema = self.predict(flownet, feature_ema)
                
                logits = (logits + logits_ema) / 2  
                outputs = (outputs + outputs_ema) / 2
                prob_flow, predicted_flow = torch.max(outputs, 1)

                total += targets.size(0)
                correct_flow += predicted_flow.eq(targets).cpu().sum().item()  
                prob_sum_flow += prob_flow.cpu().sum().item()

                if self.args.w_ce:
                    prob_ce, predicted_ce = torch.max(logits, 1)
                    correct_ce += predicted_ce.eq(targets).cpu().sum().item()  
                    prob_sum_ce += prob_ce.cpu().sum().item()

                    logits_mix = (logits + outputs) / 2
                    prob_mix, predicted_mix = torch.max(logits_mix, 1)
                    correct_mix += predicted_mix.eq(targets).cpu().sum().item()
                    prob_sum_mix += prob_mix.cpu().sum().item()
                if test_num > 0 and total >= test_num:
                    break
                    

        acc_flow = 100.*correct_flow/total
        confidence_flow = prob_sum_flow/total

        print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch, acc_flow))  
    
        if self.args.w_ce:
            acc_ce = 100.*correct_ce/total
            confidence_ce = prob_sum_ce/total
            
            acc_mix = 100.*correct_mix/total
            confidence_mix = prob_sum_mix/total

            return acc_flow, confidence_flow, acc_ce, confidence_ce, acc_mix, confidence_mix
        else:
            return acc_flow, confidence_flow

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
    def get_pseudo_label(self, net, flownet, inputs_u, inputs_u2, std = 0, updateCnetering = False):
        _, outputs_u11, features_u11 = net(inputs_u, get_feature = True)
        _, outputs_u12, features_u12 = net(inputs_u2, get_feature = True)
        
        flow_outputs_u11 = self.predict(flownet, features_u11, std, centering = (self.args.centering and updateCnetering))
        flow_outputs_u12 = self.predict(flownet, features_u12, std, centering=False)
        
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
            if self.args.w_ce:
                logMsg["pseudo/acc_net"] = acc_net
                logMsg["pseudo/confidence_net"] = confidence_net
            wandb.log(logMsg)
        return
    
    def sharpening(self, labels, T):
        if self.args.sharpening == "DINO":
            return self.sharpening_DINO(labels, T)
        elif self.args.sharpening == "UNICON":
            return self.sharpening_UNICON(labels, T)

    def sharpening_DINO(self, labels, T):
        labels_sp = labels / T
        m = nn.Softmax(dim=0)
        return m(labels_sp)

    def sharpening_UNICON(self, labels, T):
        labels = labels**(1/T)
        return labels / labels.sum(dim=1, keepdim=True)

    def entropy(self, p):
        return -p.mul((p + 1e-10).log2()).sum(dim=1)
        # b = F.softmax(x, dim=dim) * F.log_softmax(x, dim=dim)
        # e = b.sum(dim=1)
        # return e

    def log_prob(self, target, feature, flownet):
        delta_p = torch.zeros(target.shape[0], target.shape[1], 1).cuda() 
        approx21, delta_log_p2 = flownet(target, feature, delta_p)
        
        approx2 = standard_normal_logprob(approx21).view(target.size()[0], -1).sum(1, keepdim=True)
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
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # torch.distributed.all_reduce(batch_center)
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        batch_center = batch_center / (len(teacher_output))
        # ema update
        flowNet.center = flowNet.center * self.center_momentum + batch_center * (1 - self.center_momentum)