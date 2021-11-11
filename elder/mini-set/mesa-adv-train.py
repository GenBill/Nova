import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from attacker import L2PGD, LinfPGD
from dataset import Cifar10, Cifar100
from model import resnet18_small
from shades_runner import ShadesRunner
from utils import get_device_id


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
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    test_transforms = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor()
    ])

    train_dataset = Cifar10(os.environ['DATAROOT'], transform=train_transforms, train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=12, pin_memory=False)
    
    shadow_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=1)
    shadow_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=shadow_sampler, num_workers=12, pin_memory=False)

    test_dataset = Cifar10(os.environ['DATAROOT'], transform=test_transforms, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=12, pin_memory=False)

    mean = [0., 0., 0.]
    std = [1., 1., 1.]

    model = resnet18_small(n_class=train_dataset.class_num, mean=mean, std=std).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.1)
    attacker = LinfPGD(model, epsilon=8/255, step=2/255, iterations=7, random_start=True)

    criterion = nn.CrossEntropyLoss()

    runner = ShadesRunner(
        epochs, model, train_loader, shadow_loader, test_loader, 
        criterion, optimizer, scheduler, attacker, num_class, device)

    runner.mesa_train(adv=True)

    if torch.distributed.get_rank() == 0:
        torch.save(model.cpu(), './checkpoint/mesa-adv-final-cifar10.pth')
        print('Save model.')


if __name__ == '__main__':
    lr = 1e-1
    epochs = 400
    batch_size = 128
    num_class = 10
    manualSeed = 2077

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    os.environ['DATAROOT'] = '~/Datasets/cifar10'
    run(lr, epochs, batch_size, num_class)
