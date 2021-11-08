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
# from model import resnet18_small_prime as resnet18_small    # wideresnet34 as resnet18_small

from runner import EvalRunner, TargetRunner
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

    train_dataset = Cifar10(os.environ['DATAROOT'], transform=train_transforms, train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=False)
    
    test_dataset = Cifar10(os.environ['DATAROOT'], transform=test_transforms, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=False)

    model = resnet18_small(n_class=test_dataset.class_num).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
    
    for checkpoint_path in checkpoint_list:
        if torch.distributed.get_rank() == 0:
            print('\nEval on {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.state_dict())

        criterion = nn.CrossEntropyLoss()

        test_accuracy(model, test_loader)
        
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=2e-4)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[18, 20], gamma=0.1)
        # attacker = LinfPGDAttack(
        #     model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=8/255, eps_iter=2/255, nb_iter=10, 
        #     rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
        # )

        # trainer = TargetRunner(20, model, train_loader, train_loader, test_loader, criterion, optimizer, scheduler, attacker, train_dataset.class_num, device, gamma)
        # trainer.multar_train(writer, adv=True)
        # if torch.distributed.get_rank() == 0:
        #     torch.save(model.cpu(), './reTrain/'+checkpoint_path)
        #     print('Save model.')

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

        # # Test on myPGD20
        # avg_loss, acc_sum, acc_count = runner.myPGD_eval("Eval PGD10", nb_iter=10)
        # avg_loss = collect(avg_loss, runner.device)
        # avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        # if torch.distributed.get_rank() == 0:
        #     print("Eval (PGD10) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

        # # Test on myPGD20
        # avg_loss, acc_sum, acc_count = runner.myPGD_eval("Eval PGD20", nb_iter=20)
        # avg_loss = collect(avg_loss, runner.device)
        # avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        # if torch.distributed.get_rank() == 0:
        #     print("Eval (PGD20) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

        # # Test on myPGD40
        # avg_loss, acc_sum, acc_count = runner.myPGD_eval("Eval PGD40", nb_iter=40)
        # avg_loss = collect(avg_loss, runner.device)
        # avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        # if torch.distributed.get_rank() == 0:
        #     print("Eval (PGD40) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))
        
        # # Test on myPGD100
        # avg_loss, acc_sum, acc_count = runner.myPGD_eval("Eval PGD100", nb_iter=100)
        # avg_loss = collect(avg_loss, runner.device)
        # avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        # if torch.distributed.get_rank() == 0:
        #     print("Eval (PGD100) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))
        
        # # Test on myPGD200
        # avg_loss, acc_sum, acc_count = runner.myPGD_eval("Eval PGD200", nb_iter=200)
        # avg_loss = collect(avg_loss, runner.device)
        # avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        # if torch.distributed.get_rank() == 0:
        #     print("Eval (PGD200) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

        # # Test on PGD20
        avg_loss, acc_sum, acc_count = runner.PGD_eval("Eval PGD10", nb_iter=10)
        avg_loss = collect(avg_loss, runner.device)
        avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            print("Eval (PGD10) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

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

        # Test on SPSA
        avg_loss, acc_sum, acc_count = runner.SPSA_eval("Eval SPSA", nb_iter=100)
        avg_loss = collect(avg_loss, runner.device)
        avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            print("Eval (SPSA) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

        # Test on CW20 & CW100
        # avg_loss, acc_sum, acc_count = runner.CW_eval("Eval CW20", search_steps=4, nb_iter=20)
        # avg_loss = collect(avg_loss, runner.device)
        # avg_acc = collect(acc_sum, runner.device, mode='sum') / collect(acc_count, runner.device, mode='sum')
        # if torch.distributed.get_rank() == 0:
        #     print("Eval (CW20) , Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))
            
        # this_lipz = runner.Lipz_eval("Eval Lipz", nb_iter=100)
        # if torch.distributed.get_rank() == 0:
        #     print("Locally Lipz : {:.6f}".format(this_lipz))

if __name__ == '__main__':

    batch_size = 64
    manualSeed = 2049

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    writer = SummaryWriter('./runs/void')

    gamma = 0.3
    gamma_name = str(int(gamma*100))
    checkpoint_list = [
        # 'checkpoint/multar-plain-cifar10-LRDLDE.pth',
        'checkpoint/multar-plain-cifar10-LRDLDR.pth',
    ]

    os.environ['DATAROOT'] = '~/Datasets/cifar10'
    onlyeval(checkpoint_list, batch_size)

