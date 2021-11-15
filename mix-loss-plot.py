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

from runner import PtestRunner
from utils import get_device_id
from utils import collect

from count_lipz import std_lipz, adv_lipz
from advertorch.attacks import LinfPGDAttack

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def _model_freeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=False

def _model_unfreeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=True

def saveplot(x, y, filename):
    fig = plt.figure()
    plt.plot(x, y)
    fig.savefig(filename)

def onlyeval(checkpoint_list, figname_list, batch_size, num_class, writer):

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    device_id = get_device_id()
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'

    test_transforms = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor()
    ])
    
    test_dataset = Cifar10(os.environ['DATAROOT'], transform=test_transforms, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=12, pin_memory=False)

    model = resnet18_small(n_class=num_class).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
    
    for name_ii, checkpoint_path in enumerate(checkpoint_list):
        if torch.distributed.get_rank() == 0:
            print('\nEval on {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)

        _model_freeze(model)
        
        # attacker = LinfPGD(model, epsilon=8/255, step=2/255, iterations=7, random_start=True)
        # attacker = L2PGD(model, epsilon=1, step=0.1, iterations=10, random_start=True)
        attacker = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255, eps_iter=2/255, nb_iter=20, 
            rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
        )

        # 定制损失
        criterion = nn.CrossEntropyLoss(reduction='sum')
        runner = PtestRunner(model, test_loader, criterion, attacker, num_class, device)

        # count & plot
        linespace = torch.linspace(0, 1, 100)
        linespace_2 = torch.linspace(0, 2, 200)
        torch.Tensor.ndim = property(lambda self: len(self.shape))

        loss_list_0, acc_list_0, lipz_list_0 = runner.tar_adv_eval()
        loss_list_0, acc_list_0, lipz_list_0 = loss_list_0.to('cpu'), acc_list_0.to('cpu'), lipz_list_0.to('cpu')
        if torch.distributed.get_rank() == 0:
            saveplot(linespace, loss_list_0, 'figs/' + figname_list[name_ii] + '_loss_list_0.png')
            saveplot(linespace, acc_list_0, 'figs/' + figname_list[name_ii] + '_acc_list_0.png')
            saveplot(linespace, lipz_list_0, 'figs/' + figname_list[name_ii] + '_lipz_list_0.png')

        loss_list_1, acc_list_1, lipz_list_1 = runner.untar_adv_eval()
        loss_list_1, acc_list_1, lipz_list_1 = loss_list_1.to('cpu'), acc_list_1.to('cpu'), lipz_list_1.to('cpu')
        if torch.distributed.get_rank() == 0:
            saveplot(linespace, loss_list_1, 'figs/' + figname_list[name_ii] + '_loss_list_1.png')
            saveplot(linespace, acc_list_1, 'figs/' + figname_list[name_ii] + '_acc_list_1.png')
            saveplot(linespace, lipz_list_1, 'figs/' + figname_list[name_ii] + '_lipz_list_1.png')

        loss_list_2, acc_list_2, lipz_list_2 = runner.onetar_adv_eval()
        loss_list_2, acc_list_2, lipz_list_2 = loss_list_2.to('cpu'), acc_list_2.to('cpu'), lipz_list_2.to('cpu')
        if torch.distributed.get_rank() == 0:
            saveplot(linespace_2, loss_list_2, 'figs/' + figname_list[name_ii] + '_loss_list_2.png')
            saveplot(linespace_2, acc_list_2, 'figs/' + figname_list[name_ii] + '_acc_list_2.png')
            saveplot(linespace_2, lipz_list_2, 'figs/' + figname_list[name_ii] + '_lipz_list_2.png')

        loss_list_3, acc_list_3, lipz_list_3 = runner.oneuntar_adv_eval()
        loss_list_3, acc_list_3, lipz_list_3 = loss_list_3.to('cpu'), acc_list_3.to('cpu'), lipz_list_3.to('cpu')
        if torch.distributed.get_rank() == 0:
            saveplot(linespace_2, loss_list_3, 'figs/' + figname_list[name_ii] + '_loss_list_3.png')
            saveplot(linespace_2, acc_list_3, 'figs/' + figname_list[name_ii] + '_acc_list_3.png')
            saveplot(linespace_2, lipz_list_3, 'figs/' + figname_list[name_ii] + '_lipz_list_3.png')

        if torch.distributed.get_rank() == 0:
            for n_iter in range(100):
                writer.add_scalars(
                    "Rain/Loss", {
                        figname_list[name_ii]+'_0': loss_list_0[n_iter], 
                        figname_list[name_ii]+'_1': loss_list_1[n_iter], 
                    }, n_iter)
                writer.add_scalars(
                    "Rain/Acc", {
                        figname_list[name_ii]+'_0': acc_list_0[n_iter], 
                        figname_list[name_ii]+'_1': acc_list_1[n_iter], 
                    }, n_iter)
                writer.add_scalars(
                    "Rain/Lipz", {
                        figname_list[name_ii]+'_0': lipz_list_0[n_iter], 
                        figname_list[name_ii]+'_1': lipz_list_1[n_iter], 
                    }, n_iter)
            
            for n_iter in range(200):
                writer.add_scalars(
                    "Tower/Loss", {
                        figname_list[name_ii]+'_2': loss_list_2[n_iter], 
                        figname_list[name_ii]+'_3': loss_list_3[n_iter], 
                    }, n_iter)
                writer.add_scalars(
                    "Tower/Acc", {
                        figname_list[name_ii]+'_2': acc_list_2[n_iter], 
                        figname_list[name_ii]+'_3': acc_list_3[n_iter], 
                    }, n_iter)
                writer.add_scalars(
                    "Tower/Lipz", {
                        figname_list[name_ii]+'_2': lipz_list_2[n_iter], 
                        figname_list[name_ii]+'_3': lipz_list_3[n_iter], 
                    }, n_iter)

if __name__ == '__main__':

    batch_size = 512
    num_class = 10
    manualSeed = 2077
    writer = SummaryWriter('./runs/curve3')

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    checkpoint_list = [
        './checkpoint/clean-final-cifar10.pth',
        './checkpoint/adv-final-cifar10.pth',
        './checkpoint/double_tar_280.pth',
    ]

    figname_list = [
        'clean', 'adv', 'multar',
    ]

    os.environ['DATAROOT'] = '~/Datasets/cifar10'
    onlyeval(checkpoint_list, figname_list, batch_size, num_class, writer)

