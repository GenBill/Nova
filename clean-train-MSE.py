import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from attacker import L2PGD, LinfPGD
from dataset import Cifar10, Cifar10

from model import resnet18_small_prime as resnet18_small    # wideresnet34 as resnet18_small
from runner import LinfRunner as DistRunner
from utils import get_device_id, Quick_MSELoss

from advertorch.attacks import LinfPGDAttack

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
        # T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
    ])
    test_transforms = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        # T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
    ])
    # clip_min, clip_max = -1.9887, 2.1265

    train_dataset = Cifar10(os.environ['DATAROOT'], transform=train_transforms, train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=False)

    test_dataset = Cifar10(os.environ['DATAROOT'], transform=test_transforms, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=False)

    mean = [0., 0., 0.]
    std = [1., 1., 1.]

    model = resnet18_small(n_class=train_dataset.class_num, mean=mean, std=std).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
    # model = nn.parallel.DataParallel(model, device_ids=[device_id], output_device=device_id)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.1)
    # attacker = LinfPGD(model, epsilon=8/255, step=2/255, iterations=10, random_start=True)
    attacker = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=8/255, eps_iter=2/255, nb_iter=10, 
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
    )

    # criterion = nn.CrossEntropyLoss()
    criterion = Quick_MSELoss(n_class=train_dataset.class_num)

    runner = DistRunner(epochs, model, train_loader, test_loader, criterion, optimizer, scheduler, attacker, device)

    runner.train(adv=False)

    if torch.distributed.get_rank() == 0:
        torch.save(model.cpu(), './checkpoint/clean-CE-cifar10.pth')
        print('Save model.')


if __name__ == '__main__':
    lr = 1e-1
    epochs = 360
    batch_size = 128
    manualSeed = 517    # 2077

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    os.environ['DATAROOT'] = '~/Datasets/cifar10'
    run(lr, epochs, batch_size)
