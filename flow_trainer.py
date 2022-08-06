from urllib3 import Retry
import torch
import torch.nn.functional as F

from flowModule.flow import cnf
from flowModule.utils import standard_normal_logprob
import time
import sys

class FlowTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.cond_size = 128
        self.mean = 0
        self.std = 0.2
        self.sample_n = 1
        return

    def train(epoch, encoder, net, optimizer, dataloader):
        encoder.eval()
        net.train()
        num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
        
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda() 
            optimizer.zero_grad()
            
            feature, _ = encoder(inputs)
            delta_p = torch.zeros(labels.shape[0], labels.shape[1], 1).cuda() 
            approx21, delta_log_p2 = net(labels, feature, delta_p)
            
            approx2 = standard_normal_logprob(approx21).view(labels.size()[0], -1).sum(1, keepdim=True)
            delta_log_p2 = delta_log_p2.view(labels.size()[0], labels.shape[1], 1).sum(1)
            log_p2 = (approx2 - delta_log_p2)
            loss = -log_p2.mean()

            L.backward()  
            optimizer.step()                

            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | (flow) Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                    %(self.args.dataset, self.args.r, self.args.noise_mode, epoch, self.args.num_epochs, batch_idx+1, num_iter, loss.item()))
            sys.stdout.flush()

    def create_model(self):
        model = cnf(self.args.num_class, self.args.flow_modules, self.cond_size, 1).cuda()
        model = model.cuda()
        return model

    def flowSoftmax(self, approx21):
        probs = torch.clamp(approx21, min=0, max=1)
        probsSum = torch.sum(probs, 2).unsqueeze(1).expand(probs.size())
        probs /= probsSum
        return probs

    ## Calculate flow density
    def Calculate_density(self, encoder1, encoder2, model1, model2, num_samples, eval_loader):
        encoder1.eval()
        encoder2.eval()
        model1.eval()
        model2.eval()

        densitys = torch.zeros(num_samples)    
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = inputs.size()[0]

            ## Get outputs of both network
            with torch.no_grad():
                input_z1 = torch.normal(mean = self.mean, std = self.std, size=(self.sample_n * batch_size , self.args.num_class)).unsqueeze(1).cuda()
                delta_p1 = torch.zeros(input_z1.shape[0], input_z1.shape[1], 1).cuda()
                input_z2 = torch.normal(mean = self.mean, std = self.std, size=(self.sample_n * batch_size , self.args.num_class)).unsqueeze(1).cuda()
                delta_p2 = torch.zeros(input_z2.shape[0], input_z2.shape[1], 1).cuda()
                cond1, _ = encoder1(inputs)
                cond2, _ = encoder2(inputs)
                # feature1 = F.normalize(cond1, dim=1)
                # feature2 = F.normalize(cond2, dim=1)
                labels_one_hot = torch.nn.functional.one_hot(targets, self.args.num_class).type(torch.cuda.FloatTensor)
                flow_labels = labels_one_hot.unsqueeze(1).cuda()
                approx21_1, _ = model1(flow_labels, cond1, delta_p1)
                approx21_2, _ = model2(flow_labels, cond2, delta_p2)
                approx2_1 = standard_normal_logprob(approx21_1).view(flow_labels.size()[0], -1).sum(1, keepdim=True)
                approx2_2 = standard_normal_logprob(approx21_2).view(flow_labels.size()[0], -1).sum(1, keepdim=True)
                
            ## Get the Prediction
            density = (approx2_1 + approx2_2)/2     
            ## Divergence clculator to record the diff. between ground truth and output prob. density.  
            densitys[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = density.squeeze(1)

        return -densitys


    def predict(self, net, feature, mean = 0, std = 0, sample_n = 1, origin=False):
        batch_size = feature.size()[0]
        feature = F.normalize(feature, dim=1)
        feature = feature.repeat(sample_n, 1, 1)
        input_z = torch.normal(mean = mean, std = std, size=(sample_n * batch_size , self.args.num_class)).unsqueeze(1).cuda()
        delta_p = torch.zeros(input_z.shape[0], input_z.shape[1], 1).cuda()
        
        approx21, _ = net(input_z, feature, delta_p, reverse=True)
        
        probs = torch.clamp(approx21, min=0, max=1)
        probs = probs.view(sample_n, -1, self.args.num_class)
        probs_mean = torch.mean(probs, dim=0, keepdim=False)
        probs_mean = F.normalize(probs_mean, dim=1, p=1)
        if origin:
            return approx21.detach().squeeze(1)
        return probs_mean

    def testByFlow(self, net, flownet, test_loader):
        net.eval()
        flownet.eval()
        correct = 0
        total = 0
        prob_sum = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                feature, _ = net(inputs)       
                outputs = self.predict(flownet, feature, origin=True)
                # if (batch_idx == 1):
                #     print("outputs", outputs[:10])
                #     print("targets", targets[:10])
                prob, predicted = torch.max(outputs, 1)

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()  
                prob_sum += prob.cpu().sum().item()

        acc = 100.*correct/total
        ## confidence score
        confidence = prob_sum/total

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
    def get_pseudo_label(self, flownet, features_u11, features_u12):
        flow_outputs_u11 = self.predict(flownet, features_u11)
        flow_outputs_u12 = self.predict(flownet, features_u12)

        pu = (flow_outputs_u11 + flow_outputs_u12) / 2

        if(self.args.predictPolicy == "weight"):
            pu_label = torch.distributions.Categorical(pu).sample()
            pu_onehot = self.torch_onehot(pu_label, pu.shape[1]).detach()
            return pu_onehot
        else:
            # ptu = pu**(1/self.args.T)            ## Temparature Sharpening

            # targets_u = pu / pu.sum(dim=1, keepdim=True)
            # targets_u = targets_u.detach()
            return pu