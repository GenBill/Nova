import torch
import torch.nn as nn
import torch.nn.functional as F

class Quick_MSELoss(nn.Module):
    def __init__(self, n_class, reduction='mean'):
        super(Quick_MSELoss, self).__init__()
        self.n_class = n_class
        self.reduction = reduction

    def forward(self, input, label):
        target = F.one_hot(label, num_classes=self.n_class).float()
        # input = F.log_softmax(input, dim=1)
        return torch.mean(torch.norm(input-target, dim=1), dim=0)
        # return torch.mean(torch.sqrt(torch.mean((input-target)**2, dim=1)), dim=0)

class Quick_WotLoss(nn.Module):
    def __init__(self, n_class, reduction='mean'):
        super(Quick_WotLoss, self).__init__()
        self.n_class = n_class
        self.reduction = reduction

    def forward(self, input, label):
        target = F.one_hot(label, num_classes=self.n_class).float()
        # input = F.log_softmax(input, dim=1)
        return torch.norm(input-target)

class TrueQuick_MSELoss(nn.Module):
    def __init__(self, n_class, reduction='mean'):
        super(Quick_MSELoss, self).__init__()
        self.n_class = n_class
        self.reduction = reduction

    def forward(self, inputs, label):
        inputs[label] -= 1
        return torch.mean(torch.norm(inputs, dim=1), dim=0)

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