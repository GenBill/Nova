import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from attacker import L2PGD, LinfPGD
from dataset import Cifar10
from model import resnet18_small

# from runner import DistRunner
from utils import get_device_id
from utils import collect


checkpoint_path = './clean-final-cifar10.pth'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = 'cuda'

from torchvision import models
model = resnet18_small(n_class=10)
print(model)
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint["state_dict"])

m = torch.distributions.beta.Beta(1.0,1.0)
a = m.sample()
print(a)