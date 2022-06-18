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
from runner import FrostRunner # FSRunner
from utils import get_device_id, Scheduler_List, Onepixel
from utils import Quick_MSELoss, Quick_WotLoss

from advertorch.attacks import LinfPGDAttack
from attacker import DuelPGD
from tensorboardX import SummaryWriter

def run(lr, epochs, batch_size):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    device_id = get_device_id()
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'

    train_transforms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # Onepixel(32,32)
    ])
    test_transforms = T.Compose([
        T.ToTensor(),
    ])

    train_dataset = Cifar10(os.environ['DATAROOT'], transform=train_transforms, train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    test_dataset = Cifar10(os.environ['DATAROOT'], transform=test_transforms, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)

    model = resnet18_small(train_dataset.class_num).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id, )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)

    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8], gamma=1.78)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)
    # scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,220], gamma=0.5)
    scheduler = Scheduler_List([scheduler1, scheduler2])
    
    attacker_untar = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255, eps_iter=2/255, nb_iter=10, 
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
    )
    attacker_tar = LinfPGDAttack(    # DuelPGD(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255, eps_iter=2/255, nb_iter=10, 
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True, 
    )

    attacker = attacker_tar

    # criterion = nn.CrossEntropyLoss()
    criterion = Quick_MSELoss(10)
    # criterion = Quick_WotLoss(10)
    attacker_tar.loss_fn = criterion

    runner = FrostRunner(epochs, model, train_loader, test_loader, criterion, optimizer, scheduler, attacker, train_dataset.class_num, device)
    runner.eval_interval = 10
    runner.omega = 1.0
    runner.double_tar(writer)

    if torch.distributed.get_rank() == 0:
        torch.save(model.state_dict(), './check8/double_tar_Uncert10_10.pth')
        print('Save model.')

if __name__ == '__main__':
    lr = 0.1
    epochs = 280        # 320        # 240
    batch_size = 64     # 64*4 = 128*2 = 256*1
    manualSeed = 2049   # 2077

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    writer = SummaryWriter('./runs/Cifar10_double_tar')

    os.environ['DATAROOT'] = '~/Datasets/cifar10'
    run(lr, epochs, batch_size)
