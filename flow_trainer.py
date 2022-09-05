from urllib3 import Retry
import torch
import torch.nn.functional as F

import time
import sys
import torch.nn as nn

# nsf
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal, StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation, Permutation
from nflows.nn.nets import ResidualNet

class FlowTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.cond_size = 128
        self.mean = 0
        self.std = 0.2
        self.sample_n = 1
        self.flowNet_ema = None
        return

    def create_model(self, input_dim = 10, hidden_dim = 128, context_dim = 128, num_layers = 4):
        base_dist = ConditionalDiagonalNormal(shape=[input_dim], context_encoder=nn.Linear(context_dim, input_dim*2))
        # base_dist = StandardNormal(shape=[input_dim])

        transforms = []
        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=input_dim))
            transforms.append(MaskedAffineAutoregressiveTransform(features=input_dim, 
                                                                hidden_features=hidden_dim, 
                                                                context_features=context_dim,
                                                                activation= F.tanh,
                                                                use_residual_blocks=False))
        transform = CompositeTransform(transforms)

        flow = Flow(transform, base_dist)

        return flow.cuda()

    def predict(self, net, feature, mean = 0, std = 0, sample_n = 1, origin=False):
        with torch.no_grad():
            batch_size = feature.size()[0]
            feature = F.normalize(feature, dim=1)
            # feature = feature.repeat(sample_n, 1, 1)
            # input_z = torch.normal(mean = mean, std = std, size=(sample_n * batch_size , self.args.num_class)).unsqueeze(1).cuda()
            # delta_p = torch.zeros(input_z.shape[0], input_z.shape[1], 1).cuda()
            if self.args.ema:
                with self.flowNet_ema.average_parameters():
                    samples, log_prob = net.sample_and_log_prob(sample_n, feature)
            else:
                samples, log_prob  = net.sample_and_log_prob(sample_n, feature)
            # print("samples : ", samples[:10])
            probs = torch.clamp(samples, min=0, max=1)
            probs = probs.view(sample_n, -1, self.args.num_class)
            probs_mean = torch.mean(probs, dim=0, keepdim=False)
            probs_mean = F.normalize(probs_mean, dim=1, p=1)
            # print("prob : ", probs_mean[:10])
            if origin:
                return samples.detach().squeeze(1)
            return probs_mean

    def testByFlow(self, net, flownet, net_ema, test_loader):
        net.eval()
        flownet.eval()
        correct = 0
        total = 0
        prob_sum = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                if self.args.ema:
                    with net_ema.average_parameters():
                        feature, _ = net(inputs)
                else:
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
        print("prob_sum : ", prob_sum)
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
    def get_pseudo_label(self, flownet, features_u11, features_u12, std = 0):
        flow_outputs_u11 = self.predict(flownet, features_u11, std)
        flow_outputs_u12 = self.predict(flownet, features_u12, std)

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

    def setEma(self, ema):
        self.flowNet_ema = ema
        return