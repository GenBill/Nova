import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from attacker import L2PGD, LinfPGD
from dataset import Cifar10, Cifar10

from model import PreActResNet18
# from model import resnet18_small_prime as resnet18_small    # wideresnet34 as resnet18_small

from runner import TransRunner
from utils import get_device_id, test_accuracy
from utils import collect

from count_lipz import std_lipz, adv_lipz
from advertorch.attacks import LinfPGDAttack
from tensorboardX import SummaryWriter

def onlyeval(checkpoint_list, batch_size):

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
    ])

    test_transforms = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor()
    ])
    
    test_dataset = Cifar10(os.environ['DATAROOT'], transform=test_transforms, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=False)

    model_0 = PreActResNet18(num_classes=test_dataset.class_num).to(device)
    model_0 = nn.parallel.DistributedDataParallel(model_0, device_ids=[device_id], output_device=device_id)

    model_1 = PreActResNet18(num_classes=test_dataset.class_num).to(device)
    model_1 = nn.parallel.DistributedDataParallel(model_1, device_ids=[device_id], output_device=device_id)
    
    for checkpoint_path_0 in checkpoint_list:

        checkpoint_0 = torch.load(checkpoint_path_0, map_location=device)
        model_0.load_state_dict(checkpoint_0)

        for checkpoint_path_1 in checkpoint_list:
            if checkpoint_path_1 == checkpoint_path_0:
                continue
            if torch.distributed.get_rank() == 0:
                print('\nEval on {} atk {}'.format(checkpoint_path_1, checkpoint_path_0))
            checkpoint_1 = torch.load(checkpoint_path_1, map_location=device)
            model_1.load_state_dict(checkpoint_1)

            criterion = nn.CrossEntropyLoss()
            runner = TransRunner(model_0, model_1, test_dataset.class_num, test_loader, criterion, device)

            # Test on PGD20
            avg_loss, acc_sum, acc_count = runner.Trans_PGD("Trans PGD", nb_iter=20)
            avg_loss = collect(avg_loss, runner.device)
            avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
            if torch.distributed.get_rank() == 0:
                print("Trans PGD , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))


if __name__ == '__main__':

    batch_size = 128
    manualSeed = 2077

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # writer = SummaryWriter('./runs/void')

    checkpoint_list = [
        'checkpoint/adv-final-cifar10.pth',
        'checkpoint/clean-final-cifar10.pth',
    ]

    os.environ['DATAROOT'] = '~/Datasets/cifar10'
    onlyeval(checkpoint_list, batch_size)

