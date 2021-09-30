import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self) -> None:
        super(Quick_MSELoss, self).__init__()

    def forward(self, input, label):
        target = F.one_hot(label, num_classes=input.shape[-1])
        return torch.sqrt(F.mse_loss(input, target, reduction=self.reduction))
    
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