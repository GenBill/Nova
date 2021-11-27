import torch
import torch.nn as nn
from torchvision import models
import copy
import numpy as np

def _init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1 and len(m.weight.shape) > 1:
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.weight.bias)

class resnet18_rnn(nn.Module):
    def __init__(self, n_class, mean=None, std=None):
        super(resnet18_rnn, self).__init__()
        self.n_class = n_class

        # self.norm = Normalization(mean, std)
        self.encoder = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-1]+[nn.Flatten()])
        self.classifier = nn.Linear(in_features=512, out_features=n_class, bias=False)
        self.encoder.apply(_init_weight)
    
    def forward(self, x):
        # x_norm = self.norm(x)
        # f = self.encoder(x_norm)
        f = self.encoder(x)
        y = self.classifier(f)

        return y

class resnet18_small_rnn(nn.Module):
    def __init__(self, n_class, dimc=3, dimx=32, dimy=32, times=1, lamb=0.9):
        super(resnet18_small_rnn, self).__init__()
        self.n_class = n_class
        self.dimc, self.dimx, self.dimy = dimc, dimx, dimy
        self.times = times
        self.lamb = lamb

        # self.norm = Normalization(mean, std)
        self.encoder = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-1]+[nn.Flatten()])
        self.encoder[0] = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.encoder[3] = nn.Identity()
        self.classifier = nn.Linear(in_features=512, out_features=n_class, bias=False)
        self.recurrent = nn.Linear(in_features=512, out_features=dimc*dimx*dimy, bias=False)
        self.recurrent.weight.data /= 1024
        self.encoder.apply(_init_weight)
    
    def forward(self, x):
        d = x
        f = self.encoder(x)
        y = self.classifier(f)
        for _ in range(self.times-1):
            d = self.lamb*d + (1-self.lamb)*torch.clamp(self.recurrent(f).view(-1,self.dimc,self.dimx,self.dimy), 0, 1)
            f = self.encoder(d)
            y += self.classifier(f)

        return y/self.times
