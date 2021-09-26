import torch
import matplotlib.pyplot as plt


linespace = torch.linspace(0, 1, 100)
torch.Tensor.ndim = property(lambda self: len(self.shape))

yy = torch.zeros(100)

fig = plt.figure()
plt.plot(linespace, yy)
plt.plot(linespace, yy+1)
fig.savefig('figs/second.png')

# plt.imshow([0,1], [0,1])
# plt.imsave('figs/lipz_list_0.png', yy)

