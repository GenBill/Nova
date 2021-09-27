import torch

def rain_Rand(size, device):
    # rain blocker
    # return torch.rand(size,device)
    
    # /3 -> z = 1.5 -> 6.68%
    rand_n = torch.randn(size, device=device)
    rand_n = torch.abs(rand_n/3)
    rand_u = torch.rand(size, device=device)
    ret = (rand_n<=1).int()*rand_n + (rand_n>1).int()*rand_u
    return ret

def tower_Rand(size, device):
    # tower blocker
    # return torch.rand(size,device)
    
    # P0 = max，P1 = min
    rand_u = torch.rand(size, device=device) + torch.rand(size, device=device)
    return torch.abs(rand_u-1)
