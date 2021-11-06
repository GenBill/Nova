import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        # self.val = 0
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
    
    def report(self):
        return (self.sum / self.count)

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
