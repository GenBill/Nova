import torch

def my_Rand(size, device):
    rand_n = torch.randn(size, device=device)
    rand_n = torch.abs(rand_n/2)
    rand_u = torch.rand(size, device=device)
    ret = (rand_n<=1).int()*rand_n + (rand_n>1).int()*rand_u
    return ret
    
