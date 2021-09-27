import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from attacker import L2PGD, LinfPGD
from dataset import Cifar10, Cifar10
from model import resnet18_small

from runner import DistRunner
from utils import get_device_id
from utils import collect

from count_lipz import std_lipz, adv_lipz


def onlyeval(checkpoint_list, batch_size, num_class):

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
    
    for checkpoint_path in checkpoint_list:
        if torch.distributed.get_rank() == 0:
            print('\nEval on {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.state_dict())

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        attacker = LinfPGD(model, epsilon=8/255, step=2/255, iterations=10, random_start=True)
        # attacker = L2PGD(model, epsilon=1, step=0.1, iterations=10, random_start=True)

        criterion = nn.CrossEntropyLoss()

        runner = DistRunner(
            0, model, train_loader, test_loader, 
            criterion, optimizer, scheduler, attacker, device)

        # Test on Adv First
        avg_loss, acc_sum, acc_count = runner.adv_eval("Only Test")
        avg_loss = collect(avg_loss, runner.device)
        avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            print("Eval (Adver) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

        # Add Test : Adv Second
        # avg_loss, avg_true = runner.adv_eval_2("Only Test")
        # avg_loss = collect(avg_loss, runner.device)
        # avg_true = collect(avg_true, runner.device)
        # if torch.distributed.get_rank() == 0:
        #     print("Eval_2 (Adver) , 1st-True {:.6f}, True. {:.6f}".format(avg_loss, avg_true))

        # Test on Clean First
        avg_loss, acc_sum, acc_count = runner.clean_eval("Only Test")
        avg_loss = collect(avg_loss, runner.device)
        avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            print("Eval (Clean) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

        # Add Test : Clean Second
        # avg_loss, avg_true = runner.clean_eval_2("Only Test")
        # avg_loss = collect(avg_loss, runner.device)
        # avg_true = collect(avg_true, runner.device)
        # if torch.distributed.get_rank() == 0:
        #     print("Eval_2 (Clean) , 1st-True {:.6f}, True. {:.6f}".format(avg_loss, avg_true))
        
        # this_lipz = std_lipz(model, test_loader_one, device, rand_times=64, eps=8/255)
        # if torch.distributed.get_rank() == 0:
        #     print("Locally Lipz : {:.6f}".format(this_lipz))
            
        this_lipz = adv_lipz(model, test_loader, attacker, device)
        if torch.distributed.get_rank() == 0:
            print("Locally Lipz : {:.6f}".format(this_lipz))

if __name__ == '__main__':

    batch_size = 128
    num_class = 10
    manualSeed = 2077

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    gamma = 0.3
    gamma_name = str(int(gamma*100))
    checkpoint_list = [
        # './checkpoint/clean-final-cifar10.pth',
        # './checkpoint/adv-final-cifar10.pth',
        # './towerain/multar-0-cifar10.pth',
        # './towerain/target-0-cifar10.pth',
        # './towerain/target-10-cifar10.pth',
        './checkpoint/multar-0-cifar10.pth',
        './checkpoint/multar-10-cifar10.pth',
    ]

    os.environ['DATAROOT'] = '~/Datasets/cifar10'
    onlyeval(checkpoint_list, batch_size, num_class)

