import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import datasets
from tqdm.auto import tqdm

from attacker import L2PGD, LinfPGD
from dataset import Cifar10

from model import resnet18_small
from runner import FrostRunner, LinfRunner as DistRunner
from utils import get_device_id, Scheduler_List, Onepixel
from utils import Quick_MSELoss, Quick_WotLoss

from advertorch.attacks import LinfPGDAttack
from attacker import DuelPGD
from tensorboardX import SummaryWriter

import medmnist
from medmnist import INFO, Evaluator

data_flag = 'pathmnist'
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])
# # load the data
# train_dataset = DataClass(split='train', transform=test_transforms, download=download)
# test_dataset = DataClass(split='test', transform=test_transforms, download=download)
# pil_dataset = DataClass(split='train', download=download)

# # encapsulate data into dataloader form
# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# train_loader_at_eval = DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
# test_loader = DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

def run(lr, epochs, batch_size):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    device_id = get_device_id()
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'

    train_transforms = T.Compose([
        # T.RandomCrop(32, padding=4),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        # Onepixel(32,32)
    ])
    test_transforms = T.Compose([
        T.ToTensor(),
    ])

    train_dataset = DataClass(split='train', transform=test_transforms, download=download)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    test_dataset = DataClass(split='test', transform=test_transforms, download=download)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)

    model = resnet18_small(n_classes).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id, )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,120], gamma=0.1)
    
    attacker = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=8/255, eps_iter=2/255, nb_iter=10, 
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
    )

    criterion = nn.CrossEntropyLoss()

    runner = DistRunner(epochs, model, train_loader, test_loader, criterion, optimizer, scheduler, attacker, device)
    runner.train(adv=True)

    if torch.distributed.get_rank() == 0:
        torch.save(model.state_dict(), './checkpoint/MedMSE/adv-final.pth')
        print('Save model.')

if __name__ == '__main__':
    lr = 0.01
    epochs = 140        # 320        # 240
    batch_size = 32     # 64*4 = 128*2 = 256*1
    manualSeed = 2049   # 2077

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    writer = SummaryWriter('./runs/cifar10_double_tar')

    os.environ['DATAROOT'] = '~/Datasets'
    run(lr, epochs, batch_size)
