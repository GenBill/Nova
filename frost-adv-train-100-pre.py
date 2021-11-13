import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import datasets
from tqdm.auto import tqdm

from attacker import L2PGD, LinfPGD
from dataset import Cifar100

from model import PreActResNet18
from runner import FrostRunner, LinfRunner
from utils import get_device_id, Quick_MSELoss, Scheduler_List, Onepixel

from advertorch.attacks import LinfPGDAttack
from attacker import LinfPGDTargetAttack as LinfTarget
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
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        Onepixel(32,32)
    ])
    test_transforms = T.Compose([
        T.ToTensor(),
    ])

    train_dataset = Cifar100(os.environ['DATAROOT'], transform=train_transforms, train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    test_dataset = Cifar100(os.environ['DATAROOT'], transform=test_transforms, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)

    model = PreActResNet18(num_classes=train_dataset.class_num).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id, )
        # find_unused_parameters = True, broadcast_buffers = False)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)

    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,10,100], gamma=1.78)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[560,580], gamma=0.5)
    scheduler = Scheduler_List([scheduler1, scheduler2, scheduler3])
    
    attacker_untar = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255, eps_iter=2/255, nb_iter=10, 
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
    )
    attacker_tar = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255, eps_iter=2/255, nb_iter=10, 
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True, 
    )

    attacker = attacker_tar

    # criterion = nn.CrossEntropyLoss()
    criterion = Quick_MSELoss(100)

    runner = LinfRunner(100, model, train_loader, test_loader, criterion, optimizer, scheduler, attacker, device, num_class=train_dataset.class_num)
    runner.train(adv=False)

    if torch.distributed.get_rank() == 0:
        torch.save(model.state_dict(), './checkpoint/cifar100_double_tar.pth')
        print('Save model 100.')
    
    runner = FrostRunner(epochs-200, model, train_loader, test_loader, criterion, optimizer, scheduler, attacker, train_dataset.class_num, device)
    runner.double_tar(writer)

    if torch.distributed.get_rank() == 0:
        torch.save(model.state_dict(), './checkpoint/cifar100_double_tar.pth')
        print('Save model 500.')

    runner = FrostRunner(100, model, train_loader, test_loader, criterion, optimizer, scheduler, attacker, train_dataset.class_num, device)
    runner.double_tar(writer)

    if torch.distributed.get_rank() == 0:
        torch.save(model.state_dict(), './checkpoint/cifar100_double_tar.pth')
        print('Save model 600.')

if __name__ == '__main__':
    lr = 0.032*1.2
    epochs = 600
    batch_size = 64     # 64*4 = 128*2 = 256*1
    manualSeed = 2049   # 2077

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    writer = SummaryWriter('./runs/cifar100_double_tar')

    os.environ['DATAROOT'] = '~/Datasets/cifar100'
    run(lr, epochs, batch_size)
