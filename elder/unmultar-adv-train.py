import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from attacker import L2PGD, LinfPGD
from dataset import Cifar100

from model import resnet18_small    # wideresnet34 as 
from runner import unTargetRunner as TargetRunner
from utils import get_device_id, Quick_MSELoss

from advertorch.attacks import LinfPGDAttack
from tensorboardX import SummaryWriter

def run(lr, epochs, batch_size, gamma=0.5):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    device_id = get_device_id()
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'

    train_transforms = T.Compose([
        # T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    test_transforms = T.Compose([
        # T.Resize((32, 32)),
        T.ToTensor(),
   ])

    train_dataset = Cifar100(os.environ['DATAROOT'], transform=train_transforms, train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=False)

    shadow_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=1)
    shadow_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=shadow_sampler, num_workers=4, pin_memory=False)

    test_dataset = Cifar100(os.environ['DATAROOT'], transform=test_transforms, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=False)

    model = resnet18_small(n_class=train_dataset.class_num).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id, )
        # find_unused_parameters = True, broadcast_buffers = False)
    # model = nn.parallel.DataParallel(model, device_ids=[device_id], output_device=device_id)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4, alpha=0.99, eps=1e-08, centered=False)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.1)
    # attacker = LinfPGD(model, epsilon=8/255, step=2/255, iterations=10, random_start=True)
    attacker = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=8/255, eps_iter=2/255, nb_iter=10, 
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
    )

    criterion = nn.CrossEntropyLoss()
    # criterion = Quick_MSELoss()

    runner = TargetRunner(epochs, model, train_loader, shadow_loader, test_loader, criterion, optimizer, scheduler, attacker, train_dataset.class_num, device, gamma)
    runner.multar_train(writer, adv=True)

    if torch.distributed.get_rank() == 0:
        gamma_name = str(int(gamma*100))
        # torch.save(model.cpu(), './checkpoint/multar-targetmix-'+ gamma_name +'-cifar100.pth')
        torch.save(model.cpu(), './checkpoint/unmultar-softower25-cifar100.pth')
        print('Save model.')

if __name__ == '__main__':
    lr = 4e-1
    epochs = 360
    batch_size = 128
    manualSeed = 2077    # 2077
    gamma = 0.

    # writer = SummaryWriter('./runs/curve_targetmix')
    writer = SummaryWriter('./runs/void')
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    os.environ['DATAROOT'] = '~/Datasets/cifar100'
    run(lr, epochs, batch_size, gamma)
