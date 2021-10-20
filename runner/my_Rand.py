import torch

## Blocker
def brain_Rand(size, device):
    # rain blocker
    return torch.rand(size,device=device)

def btower_Rand(size, device):
    # rain blocker
    return torch.rand(size,device=device)

## std - Rand
def rain_Rand(size, device):
    # zen rain
    rand_n = torch.randn(size, device=device)
    flag = torch.abs(rand_n)
    rand_n = 1 + rand_n/2
    rand_u = torch.rand(size, device=device)
    ret = (flag<=1).int()*rand_n + (flag>1).int()*rand_u
    return ret

def noRand_ones(size, device):
    # P0 = max，P1 = min
    return torch.ones(size, device=device)

def tower_Rand(size, device):
    # P0 = max，P1 = min
    rand_u = torch.rand(size, device=device) + torch.rand(size, device=device)
    return torch.abs(rand_u-1)

def doubletower_Rand(size, device):
    # P0 = max，P1 = min
    rand_u = 2*torch.rand(size, device=device)
    return rand_u

def longtower_Rand(size, device):
    # P0 = max，P1 = min
    rand_u = torch.rand(size, device=device) + torch.rand(size, device=device)
    return torch.abs(rand_u-1)*1.2

def softower_Rand(size, device):
    # P0 = max，P1 = min
    rand_u = torch.rand(size, device=device) + torch.rand(size, device=device)
    rand_u2 = torch.rand(size, device=device)
    rand_u3 = torch.rand(size, device=device)
    ret = torch.abs(rand_u-1)*(rand_u2>0.25) + rand_u3*(rand_u2<=0.25)
    # ret = torch.abs(rand_u-1)*(rand_u2>0.125) + rand_u3*(rand_u2<=0.125)
    return ret

## inverse - Rand
def irain_Rand(size, device):
    # /3 -> z = 1.5 -> 6.68%
    rand_n = torch.randn(size, device=device)
    rand_n = torch.abs(rand_n/3)
    rand_u = torch.rand(size, device=device)
    ret = (rand_n<=1).int()*rand_n + (rand_n>1).int()*rand_u
    return 1-ret

def itower_Rand(size, device):
    # P0 = max，P1 = min
    rand_u = torch.rand(size, device=device) + torch.rand(size, device=device)
    return 1-torch.abs(rand_u-1)

def itower_Rand(size, device):
    # P0 = max，P1 = min
    rand_u = torch.rand(size, device=device) + torch.rand(size, device=device)
    return 1-torch.abs(rand_u-1)