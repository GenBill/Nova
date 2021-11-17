import torch
import torch.nn as nn
import numpy as np

def _generate_opt_means(C, p, L): 
    """
    input
        C = constant value
        p = dimention of feature vector
        L = class number
    """
    opt_means = np.zeros((L, p))
    opt_means[0][0] = 1
    for i in range(1,L):
        for j in range(i): 
            opt_means[i][j] = - (1/(L-1) + np.dot(opt_means[i],opt_means[j])) / opt_means[j][j]
        opt_means[i][i] = np.sqrt(1 - np.linalg.norm(opt_means[i])**2)
    # for k in range(L):
    #     opt_means[k] = C * opt_means[k]
    opt_means = C * opt_means
    return opt_means

class MM_LDA(nn.Module):
    def __init__(self, C, n_dense, class_num, device, Normalize=False):
        super().__init__()
        """
        C : hyperparam for generating MMD
        n_dense : dimention of the feature vector
        class_num : dimention of classes
        """
        self.C = C
        self.class_num = class_num
        opt_means = _generate_opt_means(C, n_dense, class_num)
        self.mean_expand = torch.tensor(opt_means).unsqueeze(0).double().to(device)     # (1, num_class, num_dense)
        self.Normalize = Normalize
        
    def forward(self, x):
        # print('x', x, x.shape)
        b, p = x.shape      # batch_size, num_dense
        L = self.class_num
        if self.Normalize:  # 正規化する
            x = (x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-10)) * self.C
        # print(torch.norm(x, p=2, dim=1, keepdim=True))
            
        x_expand =  x.repeat(1,L).view(b, L, p).double()                                # (batch_size, num_class, num_dense)

        logits = - torch.sum((x_expand - self.mean_expand)**2, dim=2)                   # (batch_size, num_class)
        # print('x-mean',x_expand - self.mean_expand)
        # print('logits', logits, logits.shape)
 
        return logits

class MMC_Loss(nn.Module):
    def __init__(self, Cmm, n_dense, class_num, device):
        super().__init__()
        """
        C : hyperparam for generating MMD
        n_dense : dimention of the feature vector
        class_num : dimention of classes
        """
        self.C = Cmm                                              # 固定点的模长
        self.class_num = class_num
        opt_means = _generate_opt_means(Cmm, n_dense, class_num)
        self.mean_expand = torch.tensor(opt_means).to(device)   # (num_class, num_dense)
        
    def forward(self, logits, label):
        # target = self.mean_expand[label, :]
        # loss = torch.mean(torch.norm(logits-target, dim=1), dim=0)
        return torch.mean(torch.norm(logits-self.mean_expand[label, :], dim=1), dim=0)