import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import pickle
import numpy as np
from tqdm import tqdm

from autoattack import AutoAttack

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
    
    def report(self):
        return (self.sum / self.count)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def collect(x, device, mode='mean'):
    xt = torch.tensor([x]).to(device)
    torch.distributed.all_reduce(xt, op=torch.distributed.ReduceOp.SUM)
    # print(xt.item())
    xt = xt.item()
    if mode == 'mean':
        xt /= torch.distributed.get_world_size()
    return xt
    
def get_device_id():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args.local_rank

class Quick_MSELoss(nn.Module):
    def __init__(self, n_class, reduction='mean'):
        super(Quick_MSELoss, self).__init__()
        self.n_class = n_class
        self.reduction = reduction

    def forward(self, input, label):
        target = F.one_hot(label, num_classes=self.n_class).float()
        # input = F.log_softmax(input, dim=1)
        return torch.mean(torch.norm(input-target, dim=1), dim=0)
        # return torch.mean(torch.sqrt(torch.mean((input-target)**2, dim=1)), dim=0)

    
class softCrossEntropy(nn.Module):
    def __init__(self, reduce=True):
        super(softCrossEntropy, self).__init__()
        self.reduce = reduce
        return

    def forward(self, inputs, targets):
        """
        :param inputs: predictions
        :param targets: target labels in vector form
        :return: loss
        """
        log_likelihood = -F.log_softmax(inputs, dim=1)
        sample_num, class_num = targets.shape
        if self.reduce:
            loss = torch.sum(torch.mul(log_likelihood, targets)) / sample_num
        else:
            loss = torch.sum(torch.mul(log_likelihood, targets), 1)

        return loss

class CE2MSE_Loss(nn.Module):
    def __init__(self, n_class, reduction='mean', diter=1e-4):
        super(CE2MSE_Loss, self).__init__()
        self.n_class = n_class
        self.reduction = reduction
        self.lamb = 0.
        self.diter = diter

    def forward(self, input, label):
        if self.lamb<1:
            self.lamb += self.diter
        target = F.one_hot(label, num_classes=self.n_class).float()
        mse = torch.mean(torch.sqrt(torch.sum((input-target)**2, dim=1)), dim=0)
        # mse = torch.sqrt(F.mse_loss(input, target, reduction=self.reduction))
        
        logsoftmax = nn.LogSoftmax(dim=1)
        ce = torch.mean(torch.sum(-target * logsoftmax(input), dim=1))

        return (1-self.lamb) * ce + mse


class Scheduler_2():
    def __init__(self, scheduler1, scheduler2):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
    def step(self):
        self.scheduler1.step()
        self.scheduler2.step()

class Scheduler_3():
    def __init__(self, scheduler1, scheduler2, scheduler3):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.scheduler3 = scheduler3
    def step(self):
        self.scheduler1.step()
        self.scheduler2.step()
        self.scheduler3.step()

class Scheduler_List():
    def __init__(self, mylist):
        self.mylist = mylist

    def step(self):
        for scheduler in self.mylist:
            scheduler.step()


class Onepixel():
    def __init__(self, size_x=32, size_y=32):
        self.size_x = size_x
        self.size_y = size_y

    def __call__(self, tensor):
        x = random.randint(0, self.size_x-1)
        y = random.randint(0, self.size_y-1)

        tensor[0,x,y] = random.randint(0, 1)
        tensor[1,x,y] = random.randint(0, 1)
        tensor[2,x,y] = random.randint(0, 1)

        return tensor

class addNoise():
    def __init__(self, eps=8/255):
        self.eps = 2*eps

    def __call__(self, tensor):
        rand = (torch.rand(tensor.shape)-0.5) *self.eps
        return tensor + rand


def test_accuracy(model, testloader):
    model = pickle.loads(pickle.dumps(model))
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()

    # cifar10_mean = torch.from_numpy(np.array((0.4914, 0.4822, 0.4465))).type(torch.FloatTensor)\
    #                    .cuda()[None, :, None, None].expand(1, 3, 32, 32)
    # cifar10_mean.requires_grad = False
    # cifar10_std = torch.from_numpy(np.array((0.2471, 0.2435, 0.2616))).type(torch.FloatTensor)\
    #                   .cuda()[None, :, None, None].expand(1, 3, 32, 32)
    # cifar10_std.requires_grad = False

    cifar10_mean = torch.from_numpy(np.array((0, 0, 0))).type(torch.FloatTensor) \
                       .cuda()[None, :, None, None].expand(1, 3, 32, 32)
    cifar10_mean.requires_grad = False
    cifar10_std = torch.from_numpy(np.array((1, 1, 1))).type(torch.FloatTensor) \
                      .cuda()[None, :, None, None].expand(1, 3, 32, 32)
    cifar10_std.requires_grad = False

    # attacker = CarliniWagnerL2()
    # attacker = CarliniWagnerL2Attack(predict=Packed_Model(model, cifar10_mean, cifar10_std).cuda(),
    #                                  num_classes=10,
    #                                  learning_rate=0.01,
    #                                  binary_search_steps=1,
    #                                  max_iterations=100,
    #                                  initial_const=0.0018
    #                                  )

    attacker = AutoAttack(model,    # Packed_Model(model, mean=cifar10_mean, std=cifar10_std),
                          norm='Linf', eps=8/255, version='standard', verbose=True)

    with torch.no_grad():
        iter = tqdm(testloader, ncols=200)

        top1 = AverageMeter()
        adv_top1 = AverageMeter()
        l2 = AverageMeter()

        for data, label in iter:
            data, label = data.cuda(), label.cuda()
            clean_data = data   # normalize(data.clone().detach(), cifar10_mean, cifar10_std)
            output = model(clean_data.detach())
            top1.update(accuracy(output.detach(), label.detach())[0])
            with torch.enable_grad():
                # adv_data = attack_pgd_l2(model, data.detach(), label, 0.5, 0.125, 10, restarts=1,
                #                       lower_limit=0., upper_limit=1., mu=cifar10_mean.detach(), std=cifar10_std.detach())
                # adv_data = attacker.perturb(data, label)
                adv_data = attacker.run_standard_evaluation(data.detach(), label, bs=1024)
            # adv_data = data + adv_data

            output = model(adv_data.detach())
            adv_top1.update(accuracy(output.detach(), label.detach())[0], data.size(0))
            l2.update(torch.norm((adv_data - data).view(data.size(0), -1), dim=-1).mean().item(), data.size(0))
            iter.set_description('test : top1: {}, adv_top1: {}, l2_norm: {}'
                                .format(top1.avg, adv_top1.avg, l2.avg))
        iter.close()
