import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from attacker import L2PGD, LinfPGD
from dataset import Cifar10
from model import resnet18_small
from runner import DistRunner
from utils import get_device_id
from utils import collect

from evap.instrovap import evap
# from evap.normvap import evap
from evap.onlytest import onlytest

def run_instrovap(alpha, lr, epochs, batch_size, checkpoint_list):
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
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=12, pin_memory=False)

    test_dataset = Cifar10(os.environ['DATAROOT'], transform=test_transforms, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=12, pin_memory=False)

    mean = [0., 0., 0.]
    std = [1., 1., 1.]

    model_list = []
    for i in range(len(checkpoint_list)):
        model_list.append(resnet18_small(n_class=train_dataset.class_num, mean=mean, std=std).to(device))
        model_list[i] = nn.parallel.DistributedDataParallel(model_list[i], device_ids=[device_id], output_device=device_id)
        checkpoint = torch.load(checkpoint_list[i], map_location=device)
        model_list[i].load_state_dict(checkpoint.state_dict())

    student = resnet18_small(n_class=train_dataset.class_num, mean=mean, std=std).to(device)
    student = nn.parallel.DistributedDataParallel(student, device_ids=[device_id], output_device=device_id)
    
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.1)
    attacker = LinfPGD(student, epsilon=8/255, step=2/255, iterations=7, random_start=True)

    criterion_0 = nn.CrossEntropyLoss(reduction='none')
    criterion_1 = nn.CrossEntropyLoss()
    # Train Student
    evap(alpha, epochs, student, train_loader, criterion_0, model_list, optimizer, scheduler, device)

    # Test Student
    # student = onlytest(test_loader, criterion_1, student, device)

    if torch.distributed.get_rank() == 0:
        torch.save(student.cpu(), './checkpoint/instro'+str(len(model_list))+'-final-cifar10.pth')
        print('Save model.')
    
    runner = DistRunner(
        0, student.to(device), train_loader, test_loader, 
        criterion_1, optimizer, scheduler, attacker, device)

    avg_loss, acc_sum, acc_count = runner.adv_eval("Only Test")
    avg_loss = collect(avg_loss, runner.device)
    avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
    if torch.distributed.get_rank() == 0:
        print("Eval (Adver) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

    avg_loss, acc_sum, acc_count = runner.clean_eval("Only Test")
    avg_loss = collect(avg_loss, runner.device)
    avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
    if torch.distributed.get_rank() == 0:
        print("Eval (Clean) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

if __name__ == '__main__':
    
    # loss_0 *(1-alpha) + Sigma( *alpha)
    alpha = 0.8
    
    lr = 1e-3
    epochs = 400
    batch_size = 64
    manualSeed = 2077

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    checkpoint_list = [
        # './checkpoint/clean-final-cifar10.pth',
        './checkpoint/shades-final-cifar10.pth',
        './checkpoint/adv-final-cifar10.pth',
    ]

    os.environ['DATAROOT'] = '~/Datasets/cifar10'
    run_instrovap(alpha, lr, epochs, batch_size, checkpoint_list)
    
