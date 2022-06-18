import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import datasets
from tqdm.auto import tqdm

from attacker import L2PGD, LinfPGD
from dataset import Cifar10, CUB

from model import resnet18_small
from model.resnet import resnet18, resnet34
from runner import FSRunner
from utils import get_device_id, Scheduler_List, Onepixel
from utils import Quick_MSELoss, Quick_WotLoss

from advertorch.attacks import LinfPGDAttack
from attacker import DuelPGD
from tensorboardX import SummaryWriter

import random
import math
import numbers

import cv2
import numpy as np
class Compose:
    """Composes several transforms together.

    Args:
        transforms(list of 'Transform' object): list of transforms to compose

    """    

    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):

        for trans in self.transforms:
            img = trans(img)
        
        return img
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToCVImage:
    """Convert an Opencv image to a 3 channel uint8 image
    """

    def __call__(self, image):
        """
        Args:
            image (numpy array): Image to be converted to 32-bit floating point
        
        Returns:
            image (numpy array): Converted Image
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        image = image.astype('uint8')
            
        return image


class RandomResizedCrop:
    """Randomly crop a rectangle region whose aspect ratio is randomly sampled 
    in [3/4, 4/3] and area randomly sampled in [8%, 100%], then resize the cropped
    region into a 224-by-224 square image.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped (w / h)
        interpolation: Default: cv2.INTER_LINEAR: 
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='linear'):

        self.methods={
            "area":cv2.INTER_AREA, 
            "nearest":cv2.INTER_NEAREST, 
            "linear" : cv2.INTER_LINEAR, 
            "cubic" : cv2.INTER_CUBIC, 
            "lanczos4" : cv2.INTER_LANCZOS4
        }

        self.size = (size, size)
        self.interpolation = self.methods[interpolation]
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        h, w, _ = img.shape

        area = w * h

        for attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            target_ratio = random.uniform(*self.ratio) 

            output_h = int(round(math.sqrt(target_area * target_ratio)))
            output_w = int(round(math.sqrt(target_area / target_ratio))) 

            if random.random() < 0.5:
                output_w, output_h = output_h, output_w 

            if output_w <= w and output_h <= h:
                topleft_x = random.randint(0, w - output_w)
                topleft_y = random.randint(0, h - output_h)
                break

        if output_w > w or output_h > h:
            output_w = min(w, h)
            output_h = output_w
            topleft_x = random.randint(0, w - output_w) 
            topleft_y = random.randint(0, h - output_w)

        cropped = img[topleft_y : topleft_y + output_h, topleft_x : topleft_x + output_w]

        resized = cv2.resize(cropped, self.size, interpolation=self.interpolation)

        return resized
    
    def __repr__(self):
        for name, inter in self.methods.items():
            if inter == self.interpolation:
                inter_name = name

        interpolate_str = inter_name
        format_str = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_str += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_str += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_str += ', interpolation={0})'.format(interpolate_str)

        return format_str


class RandomHorizontalFlip:
    """Horizontally flip the given opencv image with given probability p.

    Args:
        p: probability of the image being flipped
    """
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        """
        Args:
            the image to be flipped
        Returns:
            flipped image
        """
        if random.random() < self.p:
            img = cv2.flip(img, 1)
        
        return img

class ToTensor:
    """convert an opencv image (h, w, c) ndarray range from 0 to 255 to a pytorch 
    float tensor (c, h, w) ranged from 0 to 1
    """

    def __call__(self, img):
        """
        Args:
            a numpy array (h, w, c) range from [0, 255]
        
        Returns:
            a pytorch tensor
        """
        #convert format H W C to C H W
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.float() / 255.0

        return img

class Normalize:
    """Normalize a torch tensor (H, W, BGR order) with mean and standard deviation
    
    for each channel in torch tensor:
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean: sequence of means for each channel
        std: sequence of stds for each channel
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def __call__(self, img):
        """
        Args:
            (H W C) format numpy array range from [0, 255]
        Returns:
            (H W C) format numpy array in float32 range from [0, 1]
        """        
        assert torch.is_tensor(img) and img.ndimension() == 3, 'not an image tensor'

        if not self.inplace:
            img = img.clone()

        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)
        img.sub_(mean[:, None, None]).div_(std[:, None, None])

        return img

class Resize:

    def __init__(self, resized=256, interpolation='linear'):

        methods = {
            "area":cv2.INTER_AREA, 
            "nearest":cv2.INTER_NEAREST, 
            "linear" : cv2.INTER_LINEAR, 
            "cubic" : cv2.INTER_CUBIC, 
            "lanczos4" : cv2.INTER_LANCZOS4
        }
        self.interpolation = methods[interpolation]

        if isinstance(resized, numbers.Number):
            resized = (resized, resized)
        
        self.resized = resized

    def __call__(self, img):


        img = cv2.resize(img, self.resized, interpolation=self.interpolation)

        return img

def run(lr, epochs, batch_size):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    device_id = get_device_id()
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'

    TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
    TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
    TEST_MEAN = [0.4862169586881995, 0.4998156522834164, 0.4311430419332438]
    TEST_STD = [0.23264268069040475, 0.22781080253662814, 0.26667253517177186]

    train_transforms = Compose([
        ToCVImage(),
        RandomResizedCrop(448),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(TRAIN_MEAN, TRAIN_STD)
    ])

    test_transforms = Compose([
        ToCVImage(),
        Resize(448),
        ToTensor(),
        Normalize(TEST_MEAN,TEST_STD)
    ])

    train_dataset = CUB(
        os.environ['DATAROOT'],
        train=True,
        transform=train_transforms,
        target_transform=None
    )

    test_dataset = CUB(
            os.environ['DATAROOT'],
            train=False,
            transform=test_transforms,
            target_transform=None
        )

    # train_dataset = CUB(os.environ['DATAROOT'], transform=train_transforms, train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=False)

    # test_dataset = Cifar10(os.environ['DATAROOT'], transform=test_transforms, train=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=False)

    model = resnet34(train_dataset.class_num).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id, )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 180], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)
    # scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,220], gamma=0.5)
    # scheduler = Scheduler_List([scheduler1, scheduler2])
    
    attacker_untar = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255, eps_iter=2/255, nb_iter=10, 
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
    )
    attacker_tar = LinfPGDAttack(    # DuelPGD(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255, eps_iter=2/255, nb_iter=10, 
        rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True, 
    )

    attacker = attacker_untar

    criterion = nn.CrossEntropyLoss()
    # criterion = Quick_MSELoss(10)
    # criterion = Quick_WotLoss(10)
    attacker_tar.loss_fn = criterion

    runner = FSRunner(epochs, model, train_loader, test_loader, criterion, optimizer, scheduler, attacker, train_dataset.class_num, device)
    from attacker import FeaSAttack
    runner.attacker_fs = FeaSAttack(runner.model, epsilon=16/255, step=4/255, iterations=8, clip_min=-1, clip_max=1)
    runner.eval_interval = 10
    runner.plain_fs(writer)

    runner.epochs = 40
    runner.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    runner.scheduler = torch.optim.lr_scheduler.MultiStepLR(runner.optimizer, milestones=[5, 25, 35], gamma=0.1)
    runner.clean_finetune(writer)

    if torch.distributed.get_rank() == 0:
        torch.save(model.state_dict(), './checkYulan/cub_fs_3.pth')
        print('Save model.')


if __name__ == '__main__':
    lr = 0.1
    epochs = 200       # 320        # 240
    batch_size = 64    # 64*4 = 128*2 = 256*1
    manualSeed = 2049   # 2077

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    writer = SummaryWriter('./runs/void')

    os.environ['DATAROOT'] = '/home/dataset_shared/CUB/CUB_200_2011/CUB_200_2011/'
    run(lr, epochs, batch_size)
