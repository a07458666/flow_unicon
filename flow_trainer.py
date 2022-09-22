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
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

class FlowTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.cond_size = 128
        self.mean = 0
        self.std = 0.2
        self.sample_n = 1
        self.warm_up = args.warm_up
        self.contrastive_criterion = SupConLoss()
        
        ## CE Loss Functions
        if self.args.w_ce:
            self.CE       = nn.CrossEntropyLoss(reduction='none')
            self.CEloss   = nn.CrossEntropyLoss()
            self.MSE_loss = nn.MSELoss(reduction= 'none')
            self.criterion  = SemiLoss()
        if self.args.noise_mode=='asym':
            self.conf_penalty = NegEntropy()
        return

    ## For Standard Training 
    def warmup_standard(self, epoch, net, flownet, optimizer, optimizerFlow, dataloader):
        flownet.train()
        if self.args.fix == 'net':
            net.eval()
        else:    
            net.train()
        num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
        
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
            inputs, labels = inputs.cuda(), labels.cuda() 
            labels_one_hot = torch.nn.functional.one_hot(labels, self.args.num_class).type(torch.cuda.FloatTensor)

            if self.args.warmup_mixup:
                mixed_input, mixed_target = mix_match(inputs, labels_one_hot, self.args.alpha_warmup)
                _, outputs, feature_flow = net(mixed_input, get_feature = True)
                flow_labels = mixed_target.unsqueeze(1).cuda()
            else:  
                _, outputs, feature_flow = net(inputs, get_feature = True)
                flow_labels = labels_one_hot.unsqueeze(1).cuda()

            logFeature(feature_flow)            

            # == flow ==
            feature_flow = F.normalize(feature_flow, dim=1)

            delta_p = torch.zeros(flow_labels.shape[0], flow_labels.shape[1], 1).cuda()
            approx21, delta_log_p2 = flownet(flow_labels, feature_flow, delta_p)
            
            approx2 = standard_normal_logprob(approx21).view(flow_labels.size()[0], -1).sum(1, keepdim=True)
            delta_log_p2 = delta_log_p2.view(flow_labels.size()[0], flow_labels.shape[1], 1).sum(1)
            log_p2 = (approx2 - delta_log_p2)
            loss_nll = -log_p2.mean()
            # == flow end ===

            if self.args.w_ce:
                loss_ce = self.CEloss(outputs, labels)
                if self.args.noise_mode=='asym':     # Penalize confident prediction for asymmetric noise
                    penalty = self.conf_penalty(outputs)
                    L = loss_ce + penalty + loss_nll
                else:   
                    L = loss_ce + loss_nll
            else:
                L = loss_nll

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

            if self.args.ema:
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
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t NLL-loss: %.4f'
                    %(self.args.dataset, self.args.ratio, self.args.noise_mode, epoch, self.args.num_epochs, batch_idx+1, num_iter, loss_nll.item()))
            sys.stdout.flush()

    def train(self, epoch, net, flownet, optimizer, optimizerFlow, labeled_trainloader, unlabeled_trainloader):
        net.train()
        flownet.train()

        unlabeled_train_iter = iter(unlabeled_trainloader)    
        num_iter = (len(labeled_trainloader.dataset)//self.args.batch_size)+1

        for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x, labels_x_o) in enumerate(labeled_trainloader):      
            try:
                inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u_o = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u_o = unlabeled_train_iter.next()
            
            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, self.args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
            w_x = w_x.view(-1,1).type(torch.FloatTensor) 

            inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()
            
            labels_x_o = labels_x_o.cuda()
            labels_u_o = labels_u_o.cuda()

            with torch.no_grad():
                # Label co-guessing of unlabeled samples
                if self.args.ema:
                    with self.net_ema.average_parameters():
                        _, outputs_u11, features_u11_flow = net(inputs_u, get_feature = True)
                        _, outputs_u12, features_u12_flow = net(inputs_u2, get_feature = True)    
                else:
                    _, outputs_u11, features_u11_flow = net(inputs_u, get_feature = True)
                    _, outputs_u12, features_u12_flow = net(inputs_u2, get_feature = True)

                ## Pseudo-label
                pu_flow = self.get_pseudo_label(flownet, features_u11_flow, features_u12_flow, std = self.args.pseudo_std)

                pu_net = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2
                self.log_pu(pu_flow, pu_net, labels_u_o, epoch)

                if self.args.w_ce:
                    pu_net_sp = self.sharpening(pu_net, self.args.Tu)
                    targets_u = (pu_flow + pu_net_sp) / 2
                else:
                    pu = pu_flow
                    targets_u = pu
                    # targets_u = self.sharpening(pu, self.args.Tu)
                
                # lamb_Tu = (1 - linear_rampup(epoch+batch_idx/num_iter, self.warm_up, self.args.lambda_p, self.args.Tu))

                # ptu = pu**(1/lamb_Tu)            ## Temparature Sharpening
                
                # targets_u = ptu / ptu.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()                  

                ## Label refinement
                if self.args.ema:
                    with self.net_ema.average_parameters():
                        _, outputs_x1, features_x_flow  = net(inputs_x, get_feature = True)
                        _, outputs_x2, features_x2_flow = net(inputs_x2, get_feature = True)
                else:
                    _, outputs_x1, features_x_flow = net(inputs_x, get_feature = True)
                    _, outputs_x2, features_x2_flow = net(inputs_x2, get_feature = True)

                px_flow = self.get_pseudo_label(flownet, features_x_flow, features_x2_flow)
                px_net = (torch.softmax(outputs_x1, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                if self.args.w_ce:
                    px = (px_flow + px_net) / 2
                else:
                    px = px_flow

                px_mix = w_x*labels_x + (1-w_x)*px

                # ptx = px_mix**(1/self.args.Tx)    ## Temparature sharpening 
                            
                # targets_x = ptx / ptx.sum(dim=1, keepdim=True)   
                targets_x = self.sharpening(px_mix, self.args.Tu)        
                targets_x = targets_x.detach()

                self.print_label_status(targets_x, targets_u, labels_x_o, labels_u_o, epoch)

                logFeature(torch.cat([features_u11_flow, features_u12_flow, features_x_flow, features_x2_flow], dim=0))
                # Calculate label sources
                u_sources_pseudo = js_distance(targets_u, labels_u_o, self.args.num_class)
                x_sources_origin = js_distance(labels_x, labels_x_o, self.args.num_class)
                x_sources_refine = js_distance(targets_x, labels_x_o, self.args.num_class)

            ## Unsupervised Contrastive Loss
            f1, _ = net(inputs_u3)
            f2, _ = net(inputs_u4)
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
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
                Lx, Lu, lamb = self.criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, self.warm_up)
                
                ## Regularization
                prior = torch.ones(self.args.num_class)/self.args.num_class
                prior = prior.cuda()        
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior*torch.log(prior/pred_mean))
                loss_unicon = Lx + lamb * Lu + penalty

            ## Flow loss
            flow_feature = F.normalize(flow_feature, dim=1)
            flow_mixed_target = mixed_target.unsqueeze(1).cuda()
            delta_p = torch.zeros(flow_mixed_target.shape[0], flow_mixed_target.shape[1], 1).cuda() 
            approx21, delta_log_p2 = flownet(flow_mixed_target, flow_feature, delta_p)
            
            approx2 = standard_normal_logprob(approx21).view(mixed_target.size()[0], -1).sum(1, keepdim=True)
            delta_log_p2 = delta_log_p2.view(flow_mixed_target.size()[0], flow_mixed_target.shape[1], 1).sum(1)
            log_p2 = (approx2 - delta_log_p2)

            lamb_u = linear_rampup(epoch+batch_idx/num_iter, self.warm_up, self.args.linear_u, self.args.lambda_u) + self.args.lamb_u_base
            
            loss_nll_x = -log_p2[:batch_size*2].mean()
            loss_nll_u = -log_p2[batch_size*2:].mean()
            loss_flow = loss_nll_x + lamb_u * loss_nll_u

            ## Total Loss
            if self.args.w_ce:
                loss = loss_unicon + self.args.lambda_c * loss_simCLR + loss_flow + reg_f_var_loss
            else:
                loss = self.args.lambda_c * loss_simCLR + loss_flow + reg_f_var_loss

            # Compute gradient and Do SGD step
            optimizer.zero_grad()
            optimizerFlow.zero_grad()
            loss.backward()

            if self.args.fix == 'flow':
                optimizer.step()
            elif self.args.fix == 'net':
                optimizerFlow.step()  
            else:
                optimizer.step()
                optimizerFlow.step()  

            if self.args.ema:
                self.net_ema.update()
                self.flowNet_ema.update()

            ## wandb
            if (wandb != None):
                logMsg = {}
                logMsg["epoch"] = epoch
                # logMsg["lamb_Tu"] = lamb_Tu
                
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

                logMsg["label_quality/unlabel_pseudo_JSD_mean"] = u_sources_pseudo.mean().item()
                logMsg["label_quality/label_origin_JSD_mean"] = x_sources_origin.mean().item()
                logMsg["label_quality/label_refine_JSD_mena"] = x_sources_refine.mean().item()

                if self.args.w_ce:
                    logMsg["loss/ce_x"] = Lx.item()
                    logMsg["loss/mse_u"] = Lu.item()
                    logMsg["loss/penalty"] = penalty.item()

                wandb.log(logMsg)

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Contrastive Loss:%.4f NLL(x) loss: %.2f NLL(u) loss: %.2f'
                %(self.args.dataset, self.args.ratio, self.args.noise_mode, epoch, self.args.num_epochs, batch_idx+1, num_iter, loss_simCLR.item(),  loss_nll_x.mean().item(), loss_nll_u.mean().item()))
        sys.stdout.flush()
        

    ## Calculate JSD
    def Calculate_JSD(self, net, flowNet, num_samples, eval_loader):  
        JSD   = torch.zeros(num_samples)    
        for batch_idx, (inputs, targets, index) in tqdm(enumerate(eval_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = inputs.size()[0]

            ## Get outputs of both network
            with torch.no_grad():
                if self.args.ema:
                    with self.net_ema.average_parameters():
                        _, _, feature = net(inputs, get_feature=True)
                else:
                    _, _, feature  = net(inputs, get_feature=True)
                feature = F.normalize(feature, dim=1)
                out = self.predict(flowNet, feature)

            ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
            dist = js_distance(out, targets, self.args.num_class)
            JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist
        return JSD

    def create_model(self):
        model = cnf(self.args.num_class, self.args.flow_modules, self.cond_size, 1).cuda()
        model = model.cuda()
        return model

    def predict(self, flowNet, feature, mean = 0, std = 0, sample_n = 1, origin=False):
        with torch.no_grad():
            batch_size = feature.size()[0]
            feature = F.normalize(feature, dim=1)
            feature = feature.repeat(sample_n, 1, 1)
            input_z = torch.normal(mean = mean, std = std, size=(sample_n * batch_size , self.args.num_class)).unsqueeze(1).cuda()
            delta_p = torch.zeros(input_z.shape[0], input_z.shape[1], 1).cuda()
            if self.args.ema:
                with self.flowNet_ema.average_parameters():
                    approx21, _ = flowNet(input_z, feature, delta_p, reverse=True)
            else:
                approx21, _ = flowNet(input_z, feature, delta_p, reverse=True)
            probs = torch.clamp(approx21, min=0, max=1)
            probs = probs.view(sample_n, -1, self.args.num_class)
            probs_mean = torch.mean(probs, dim=0, keepdim=False)
            probs_mean = F.normalize(probs_mean, dim=1, p=1)
            if origin:
                return approx21.detach().squeeze(1)
            return probs_mean

    def testByFlow(self, epoch, net, flownet, test_loader):
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
                if self.args.ema:
                    with self.net_ema.average_parameters():
                        _, logits, feature = net(inputs, get_feature = True)
                else:
                       _, logits, feature = net(inputs, get_feature = True)

                outputs = self.predict(flownet, feature, origin=True)
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
    def get_pseudo_label(self, flownet, features_u11, features_u12, std = 0):
        flow_outputs_u11 = self.predict(flownet, features_u11, std)
        flow_outputs_u12 = self.predict(flownet, features_u12, std)

        pu = (flow_outputs_u11 + flow_outputs_u12) / 2
        
        return pu
        # pu_label = torch.distributions.Categorical(pu).sample()
        # pu_onehot = self.torch_onehot(pu_label, pu.shape[1]).detach()
        # return pu_onehot

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
        labels = labels**(1/T)
        return labels / labels.sum(dim=1, keepdim=True)