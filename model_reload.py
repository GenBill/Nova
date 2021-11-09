import os
import random

import torch
import torch.nn as nn

from model import resnet18_small
from utils import get_device_id
from utils import collect

def model_reload(checkpoint_list):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    device_id = get_device_id()
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'

    model = resnet18_small(n_class=10).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)

    for checkpoint_path in checkpoint_list:
        if torch.distributed.get_rank() == 0:
            print('\nEval on {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.state_dict())

        if torch.distributed.get_rank() == 0:
            torch.save(model.state_dict(), './checkpoint/dict/multar-plain-cifar10-LRDLDEDSL2-8421-save.pth')
            print('Save model.')

if __name__ == '__main__':
    checkpoint_list = [
        'checkpoint/multar-plain-cifar10-LRDLDEDSL2-8421.pth',
    ]
    model_reload(checkpoint_list)