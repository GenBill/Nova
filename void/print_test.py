import torch
import torch.nn as nn

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




pred = torch.rand(2,4)
print(pred)

softmax_pred = nn.LogSoftmax(dim=1)(pred)
print(softmax_pred)

import random
print(random.randint(0,1))
print(random.randint(0,1))
print(random.randint(0,1))