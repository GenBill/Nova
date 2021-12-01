import torch
import torch.nn as nn

a = torch.rand(2,3)
b = torch.rand(2,3)
print(a, b)
loss = nn.MSELoss()

print((a-b)**2)
print(torch.mean((a-b)**2))
print(loss(a,b))