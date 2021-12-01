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

class desnet18_small(nn.Module):
    def __init__(self, n_class, mean=None, std=None):
        super(desnet18_small, self).__init__()
        self.n_class = n_class

        # self.norm = Normalization(mean, std)
        self.backbone = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-1])
        self.backbone[0] = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.backbone[3] = nn.Identity()

        self.head = self.backbone[:4]
        self.block_0 = self.backbone[4]
        self.block_1 = self.backbone[5]
        self.block_2 = self.backbone[6]
        self.block_3 = self.backbone[7]

        self.classifier = nn.Sequential(self.backbone[8], nn.Flatten(), nn.Linear(in_features=512, out_features=n_class, bias=False))
        self.backbone.apply(_init_weight)
    
    def forward(self, x, train=False):
        if train:
            x = self.head(x)
            f0 = self.block_0(x)
            f1 = self.block_1(f0)
            f2 = self.block_2(f1)
            f3 = self.block_3(f2)
            y = self.classifier(f3)
            return y, f0, f1, f2, f3
        else:
            f = self.backbone(x)
            y = self.classifier(f)
            return y