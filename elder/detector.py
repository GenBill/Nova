import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from attacker import L2PGD, LinfPGD
from dataset import Cifar10, Cifar100, ImageNet
from model import resnet18_small

from ada_runner import AdaRunner
from utils import get_device_id
from utils import collect

class Detector(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_class, num_class),
            nn.ReLU(),
            nn.Linear(num_class, num_class),
            nn.ReLU(),
            nn.Linear(num_class, 2)
        )

    def forward(self, x):
        return self.mlp(x)

def detector(checkpoint_list, batch_size, num_class):
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
    
    test_dataset = Cifar10(os.environ['DATAROOT'], transform=test_transforms, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=12, pin_memory=False)
    test_loader_one = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=2, pin_memory=False)

    mean = [0., 0., 0.]
    std = [1., 1., 1.]

    model = resnet18_small(n_class=num_class, mean=mean, std=std).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
    
    for index, checkpoint_path in enumerate(checkpoint_list):
        if torch.distributed.get_rank() == 0:
            print('\nEval on {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.state_dict())

        ada_net = Detector(num_class).to(device)
        ada_net = nn.parallel.DistributedDataParallel(ada_net, device_ids=[device_id], output_device=device_id)

        optimizer = torch.optim.SGD(ada_net.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300], gamma=0.1)
        attacker = LinfPGD(model, epsilon=8/255, step=2/255, iterations=7, random_start=True)
        # attacker = L2PGD(model, epsilon=1, step=0.1, iterations=10, random_start=True)

        criterion = nn.CrossEntropyLoss()

        runner = AdaRunner(
            0, model, ada_net, train_loader, test_loader, 
            criterion, optimizer, scheduler, attacker, device)

        # Test on Adv
        avg_loss, acc_sum, acc_count = runner.adv_eval(str(index))
        avg_loss = collect(avg_loss, runner.device)
        avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            print("Eval (Adver) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

        # Test on Clean
        avg_loss, acc_sum, acc_count = runner.clean_eval(str(index))
        avg_loss = collect(avg_loss, runner.device)
        avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            print("Eval (Clean) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))


if __name__ == '__main__':

    batch_size = 128
    num_class = 10
    manualSeed = 517    # 2077，517，6204

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    checkpoint_list = [
        './checkpoint/clean-final-cifar10.pth',
        './checkpoint/adv-final-cifar10.pth',
        './checkpoint/shades-clean-final-cifar10.pth',
        './checkpoint/shades-adv-final-cifar10.pth',
        # './checkpoint/double-adv-final-cifar10.pth',
        # './checkpoint/mesa-adv-final-cifar10.pth',
    ]

    os.environ['DATAROOT'] = '~/Datasets/cifar10'
    detector(checkpoint_list, batch_size, num_class)

