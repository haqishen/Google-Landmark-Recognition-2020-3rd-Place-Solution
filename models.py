import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck
import geffnet
from rexnetv1 import ReXNetV1
from resnest.torch import resnest101


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class CrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self, n_dim, ls_=0.9):
        super().__init__()
        self.n_dim = n_dim
        self.ls_ = ls_

    def forward(self, x, target):
        target = F.one_hot(target, self.n_dim).float()
        target *= self.ls_
        target += (1 - self.ls_) / self.n_dim

        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
            
    def forward(self, logits, labels, out_dim):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss     


class Effnet_Landmark(nn.Module):

    def __init__(self, enet_type, out_dim):
        super(Effnet_Landmark, self).__init__()
        self.enet = geffnet.create_model(enet_type.replace('-', '_'), pretrained=False)
        self.feat = nn.Linear(self.enet.classifier.in_features, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        logits_m = self.metric_classify(self.swish(self.feat(x)))
        return logits_m


class RexNet20_Landmark(nn.Module):

    def __init__(self, enet_type, out_dim, load_pretrained=True):
        super(RexNet20_Landmark, self).__init__()
        self.enet = ReXNetV1(width_mult=2.0)
        if load_pretrained:
            pretrain_wts = "/workspace/rexnetv1_2.0x.pth"            
            sd = torch.load(pretrain_wts)
            self.enet.load_state_dict(sd, strict=True)        
        
        self.feat = nn.Linear(self.enet.output[1].in_channels, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.output = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        logits_m = self.metric_classify(self.swish(self.feat(x)))
        return logits_m
        

class ResNest101_Landmark(nn.Module):

    def __init__(self, enet_type, out_dim):
        super(ResNest101_Landmark, self).__init__()
        self.enet = resnest101(pretrained=True)
        self.feat = nn.Linear(self.enet.fc.in_features, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.fc = nn.Identity()
        
    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        logits_m = self.metric_classify(self.swish(self.feat(x)))
        return logits_m
