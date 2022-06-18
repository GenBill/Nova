import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from attacker import L2PGD, LinfPGD
from dataset import Cifar100, Cifar10, SVHN

from model import resnet18_small
from runner import ShadesRunner
from utils import get_device_id, Scheduler_List, Onepixel


def run(lr, epochs, batch_size, num_class):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    device_id = get_device_id()
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'

    train_transforms = T.Compose([
        T.RandomCrop(32, padding=4),
        # T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    test_transforms = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor()
    ])

    train_dataset = SVHN(os.environ['DATAROOT'], transform=train_transforms, train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    
    shadow_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=1)
    shadow_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=shadow_sampler, num_workers=4, pin_memory=True)

    test_dataset = SVHN(os.environ['DATAROOT'], transform=test_transforms, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)

    model = resnet18_small(n_class=train_dataset.class_num).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120,140], gamma=0.1)
    # scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)
    # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120,240,360], gamma=0.1)
    # scheduler = Scheduler_List([scheduler1, scheduler2])

    attacker = LinfPGD(model, epsilon=8/255, step=2/255, iterations=10, random_start=True)

    criterion = nn.CrossEntropyLoss()

    runner = ShadesRunner(
        epochs, model, train_loader, shadow_loader, test_loader, 
        criterion, optimizer, scheduler, attacker, num_class, device)
    
    runner.omega = 0.8
    runner.Uncert_train(adv=False)

    if torch.distributed.get_rank() == 0:
        torch.save(model.state_dict(), './check4/shades-clean-final-svhn-Uce-08.pth')
        print('Save model.')


if __name__ == '__main__':
    lr = 0.01
    epochs = 150
    batch_size = 64
    num_class = 10
    manualSeed = 2077    # 2077，517，6204

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    os.environ['DATAROOT'] = '~/Datasets/svhn'
    run(lr, epochs, batch_size, num_class)
