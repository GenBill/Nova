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
from runner import FrostRunner
from utils import Quick_MSELoss, MM_LDA, MMC_Loss_ch
from utils import get_device_id, Scheduler_List, Onepixel

from advertorch.attacks import LinfPGDAttack
# from attacker import LinfPGDTargetAttack as LinfTarget
from tensorboardX import SummaryWriter

from runner import EvalRunner
from utils import get_device_id, test_accuracy
from utils import collect

from count_lipz import std_lipz, adv_lipz
from advertorch.attacks import LinfPGDAttack
from tensorboardX import SummaryWriter

def onlyeval(model, device):
    test_transforms = T.Compose([
        T.ToTensor()
    ])
    
    test_dataset = Cifar10(os.environ['DATAROOT'], transform=test_transforms, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0, pin_memory=False)

    # criterion = nn.CrossEntropyLoss()
    criterion = MMC_Loss_ch()

    # test_accuracy(model, test_loader)

    runner = EvalRunner(model, test_dataset.class_num, test_loader, criterion, device)

    # Test on Clean
    avg_loss, acc_sum, acc_count = runner.clean_eval("Eval Clean")
    avg_loss = collect(avg_loss, runner.device)
    avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
    if torch.distributed.get_rank() == 0:
        print("Eval (Clean) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

    # Test on FGSM
    avg_loss, acc_sum, acc_count = runner.FGSM_eval("Eval FGSM")
    avg_loss = collect(avg_loss, runner.device)
    avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
    if torch.distributed.get_rank() == 0:
        print("Eval (FGSM) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

    # # Test on PGD10
    # avg_loss, acc_sum, acc_count = runner.PGD_eval("Eval PGD10", nb_iter=10)
    # avg_loss = collect(avg_loss, runner.device)
    # avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
    # if torch.distributed.get_rank() == 0:
    #     print("Eval (PGD10) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

    # # Test on PGD20
    avg_loss, acc_sum, acc_count = runner.PGD_eval("Eval PGD20", nb_iter=20)
    avg_loss = collect(avg_loss, runner.device)
    avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
    if torch.distributed.get_rank() == 0:
        print("Eval (PGD20) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

    # # Test on PGD100
    avg_loss, acc_sum, acc_count = runner.PGD_eval("Eval PGD100", nb_iter=100)
    avg_loss = collect(avg_loss, runner.device)
    avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
    if torch.distributed.get_rank() == 0:
        print("Eval (PGD100) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))
        
    this_lipz = runner.Lipz_eval(nb_iter=100)
    if torch.distributed.get_rank() == 0:
        print("Locally Lipz : {:.6f}".format(this_lipz))

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
        Onepixel(32,32)
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

    model = resnet18_small(256).to(device)
    model = nn.Sequential(model, MM_LDA(1, 256, 10, device))   # MM_LDA(10, 256, 10, device))
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
    attacker_tar = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255, eps_iter=2/255, nb_iter=0, 
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True, 
    )

    attacker = attacker_tar

    # criterion = nn.CrossEntropyLoss()
    # criterion = Quick_MSELoss(10)
    criterion = MMC_Loss_ch()
    # criterion = MMC_Loss(Cmm=10, n_dense=256, class_num=10, device=device)

    runner = FrostRunner(epochs, model, train_loader, test_loader, criterion, optimizer, scheduler, attacker, train_dataset.class_num, device)
    runner.eval_interval = 10
    runner.attacker = LinfPGDAttack(
        model, loss_fn=criterion, eps=8/255, eps_iter=2/255, nb_iter=0, 
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
    )
    runner.std_attacker = LinfPGDAttack(
        model, loss_fn=criterion, eps=8/255, eps_iter=2/255, nb_iter=20, 
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
    )
    runner.mmc_vertex_untar(writer)
    # runner.mmc_vertex_untar(writer)

    if torch.distributed.get_rank() == 0:
        torch.save(model.state_dict(), './checkpoint/none_mmc.pth')
        print('Save model.')
    
    # onlyeval(model, device)

if __name__ == '__main__':
    lr = 0.0032
    epochs = 280        # 320        # 240
    batch_size = 64    # 64*4 = 128*2 = 256*1
    manualSeed = 2049   # 2077

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    writer = SummaryWriter('./runs/cifar10_double_tar')

    os.environ['DATAROOT'] = '~/Datasets/cifar10'
    run(lr, epochs, batch_size)
