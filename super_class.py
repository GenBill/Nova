from torch.utils.data import dataset
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import datasets

from dataset import Cifar100
from utils import get_device_id, Quick_MSELoss, Scheduler_List, Onepixel

train_transforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    Onepixel(32,32)
])

batch_size = 128
dataroot = '~/Datasets/cifar100'

train_dataset = Cifar100(dataroot, transform=train_transforms, train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

print(train_dataset.classes)

testlist = [1,2,3,4,5,6,7,8]*4
ind = [2,4,6,8,10]
print(testlist[ind])


