import torch

a = torch.rand(4,2)
gamma = 0.3
print(str(int(gamma*10)))

Sa = list(a.shape)
Sa[1] = 10
print(Sa)

b = torch.rand(Sa)
print(a.shape, b.shape)

label=torch.LongTensor([ [0],[1],[2] ])
print(label.shape)
