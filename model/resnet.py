import torch
import torch.nn as nn
from torchvision import models
import copy

def _init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1 and len(m.weight.shape) > 1:
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.weight.bias)

class resnet18(nn.Module):
    def __init__(self, n_class, mean=None, std=None):
        super(resnet18, self).__init__()
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

class resnet18_small(nn.Module):
    def __init__(self, n_class, mean=None, std=None):
        super(resnet18_small, self).__init__()
        self.n_class = n_class

        # self.norm = Normalization(mean, std)
        self.encoder = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-1]+[nn.Flatten()])
        self.encoder[0] = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.encoder[3] = nn.Identity()
        # self.classifier = nn.Linear(in_features=512, out_features=n_class, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=n_class, bias=False),
        )
        self.encoder.apply(_init_weight)
    
    def forward(self, x):
        # x_norm = self.norm(x)
        # f = self.encoder(x_norm)
        f = self.encoder(x)
        y = self.classifier(f)

        return y


class resnet34(nn.Module):
    def __init__(self, n_class, mean=None, std=None):
        super(resnet34, self).__init__()
        self.n_class = n_class

        # self.norm = Normalization(mean, std)
        self.encoder = nn.Sequential(*list(models.resnet34(pretrained=False).children())[:-1]+[nn.Flatten()])
        self.classifier = nn.Linear(in_features=512, out_features=n_class, bias=True)
        self.encoder.apply(_init_weight)
    
    def forward(self, x):
        x_norm = self.norm(x)
        f = self.encoder(x_norm)
        y = self.classifier(f)

        return y

class resnet34_small(nn.Module):
    def __init__(self, n_class, mean=None, std=None):
        super(resnet34_small, self).__init__()
        self.n_class = n_class

        # self.norm = Normalization(mean, std)
        self.encoder = nn.Sequential(*list(models.resnet34(pretrained=False).children())[:-1]+[nn.Flatten()])
        self.encoder[0] = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.encoder[3] = nn.Identity()
        self.classifier = nn.Linear(in_features=512, out_features=n_class, bias=True)
        self.encoder.apply(_init_weight)
    
    def forward(self, x):
        x_norm = self.norm(x)
        f = self.encoder(x_norm)
        y = self.classifier(f)

        return y

class resnet50(nn.Module):
    def __init__(self, n_class, mean=None, std=None):
        super(resnet50, self).__init__()
        self.n_class = n_class

        # self.norm = Normalization(mean, std)
        self.encoder = nn.Sequential(*list(models.resnet50(pretrained=False).children())[:-1]+[nn.Flatten()])
        self.classifier = nn.Linear(in_features=2048, out_features=n_class, bias=True)
        self.encoder.apply(_init_weight)
    
    def forward(self, x):
        x_norm = self.norm(x)
        f = self.encoder(x_norm)
        y = self.classifier(f)

        return y
