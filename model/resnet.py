import torch
import torch.nn as nn
from torchvision import models
import copy
import numpy as np
def _generate_opt_means(C, p, L): 
    """
    input
        C = constant value
        p = dimention of feature vector
        L = class number
    """
    opt_means = np.zeros((L, p))
    opt_means[0][0] = 1
    for i in range(1,L):
        for j in range(i): 
            opt_means[i][j] = - (1/(L-1) + np.dot(opt_means[i],opt_means[j])) / opt_means[j][j]
        opt_means[i][i] = np.sqrt(1 - np.linalg.norm(opt_means[i])**2)
    for k in range(L):
        opt_means[k] = C * opt_means[k]
        
    return opt_means

class MM_LDA(nn.Module):
    def __init__(self, C, n_dense, class_num, ):
        super().__init__()
        """
        C : hyperparam for generating MMD
        n_dense : dimention of the feature vector
        class_num : dimention of classes
        """
        self.C = C
        self.class_num = class_num
        opt_means = _generate_opt_means(C, n_dense, class_num)
        self.mean_expand = torch.tensor(opt_means).unsqueeze(0) # (1, num_class, num_dense)
        
    def forward(self, x):

        b, p = x.shape
        L = self.class_num
        if self.Normalize:
            x = (x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-10)) * self.C

        x_expand =  x.repeat(1,L).view(b, L, p)
        logits = - torch.sum((x_expand - self.mean_expand)**2, dim=2)
 
        return logits

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
            nn.Linear(in_features=512, out_features=256, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=256, out_features=256, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=256, out_features=n_class, bias=False),
        )
        self.encoder.apply(_init_weight)
    
    def forward(self, x):
        # x_norm = self.norm(x)
        # f = self.encoder(x_norm)
        f = self.encoder(x)
        y = self.classifier(f)

        return y

class resnet18_mmc(nn.Module):
    def __init__(self, n_class, mean=None, std=None):
        super(resnet18_mmc, self).__init__()
        self.n_class = n_class

        # self.norm = Normalization(mean, std)
        self.encoder = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-1]+[nn.Flatten()])
        self.encoder[0] = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.encoder[3] = nn.Identity()
        self.classifier = nn.Linear(in_features=512, out_features=n_class, bias=False)

        self.encoder.apply(_init_weight)
        self.mmc = MM_LDA(10, 256, n_class)
    
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
