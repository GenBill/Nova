import torch
import torch.nn as nn
from torchvision import models
import copy
import numpy as np

from advertorch.attacks import LinfPGDAttack

def _init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1 and len(m.weight.shape) > 1:
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.weight.bias)

class tenet18(nn.Module):
    def __init__(self, n_class, mean=None, std=None):
        super(tenet18, self).__init__()
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

class tenet18_small(nn.Module):
    def __init__(self, n_class, mean=None, std=None):
        super(tenet18_small, self).__init__()
        self.n_class = n_class

        # self.norm = Normalization(mean, std)
        self.encoder = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-1]+[nn.Flatten()])
        self.encoder[0] = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.encoder[3] = nn.Identity()
        self.classifier = nn.Linear(in_features=512, out_features=n_class, bias=False)
        
        self.encoder.apply(_init_weight)
        self.attacker = LinfPGDAttack(
            self, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255, eps_iter=2/255, nb_iter=10, 
            rand_init=False, clip_min=0.0, clip_max=1.0, targeted=True, 
        )
    
    def forward_plain(self, x):
        # x_norm = self.norm(x)
        # f = self.encoder(x_norm)
        f = self.encoder(x)
        y = self.classifier(f)

        return y

    def forward(self, x):
        y = 0
        for i in range(self.n_class):
            target = i*torch.ones(x.shape[0], device=x.device)
            x_tar = self.attacker.perturb(x, target)

            f_tar = self.encoder(x_tar)
            y += self.classifier(f_tar)

        return y
