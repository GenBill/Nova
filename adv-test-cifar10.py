import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from attacker import L2PGD, LinfPGD
from dataset import Cifar10, Cifar10
from model import resnet18_small as resnet18_small
# from model import resnet18_small_prime as resnet18_small

from runner import EvalRunner
from utils import get_device_id
from utils import collect

from count_lipz import std_lipz, adv_lipz


def onlyeval(checkpoint_list, batch_size):

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

    model = resnet18_small(n_class=test_dataset.class_num).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
    
    for checkpoint_path in checkpoint_list:
        if torch.distributed.get_rank() == 0:
            print('\nEval on {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.state_dict())

        criterion = nn.CrossEntropyLoss()

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

        # Test on PGD20
        avg_loss, acc_sum, acc_count = runner.PGD_eval("Eval PGD20", nb_iter=20)
        avg_loss = collect(avg_loss, runner.device)
        avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            print("Eval (PGD20) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

        # Test on PGD100
        avg_loss, acc_sum, acc_count = runner.PGD_eval("Eval PGD100", nb_iter=100)
        avg_loss = collect(avg_loss, runner.device)
        avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            print("Eval (PGD100) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

        # Test on CW20 & CW100
        # avg_loss, acc_sum, acc_count = runner.CW_eval("Eval CW20", search_steps=4, nb_iter=20)
        # avg_loss = collect(avg_loss, runner.device)
        # avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        # if torch.distributed.get_rank() == 0:
        #     print("Eval (CW20) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))
            
        this_lipz = runner.Lipz_eval(nb_iter=100)
        if torch.distributed.get_rank() == 0:
            print("Locally Lipz : {:.6f}".format(this_lipz))

if __name__ == '__main__':

    batch_size = 128
    manualSeed = 2077

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    gamma = 0.3
    gamma_name = str(int(gamma*100))
    checkpoint_list = [
        # './checkpoint/clean-prime-cifar10.pth',
        # './checkpoint/adv-prime-cifar10.pth',
        './checkpoint/Vertexmix-soft25-cifar10.pth',
        # './checkpoint/adv-final-cifar10.pth',
        # './checkpoint/multar-EdgeMix-0-cifar10.pth',
        # './checkpoint/target-pgd-MSE-cifar10.pth',
    ]

    os.environ['DATAROOT'] = '~/Datasets/cifar10'
    onlyeval(checkpoint_list, batch_size)

