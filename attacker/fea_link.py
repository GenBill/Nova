import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import ot

def _model_freeze(model):
        for param in model.parameters():
            param.requires_grad=False

def _model_unfreeze(model):
    for param in model.parameters():
        param.requires_grad=True

def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

def label_smoothing(y_batch_tensor, num_classes, delta):
    y_batch_smooth = (1 - delta - delta / (num_classes - 1)) * \
        y_batch_tensor + delta / (num_classes - 1)
    return y_batch_smooth

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

class FeaSAttack(nn.Module):
    def __init__(self, model, epsilon=8/255, step=2/255, iterations=20, clip_min=0.0, clip_max=1.0):
        super(FeaSAttack, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.epsilon = epsilon
        self.step = step
        self.iterations = iterations

        self.clip_min = clip_min
        self.clip_max = clip_max

        # Model status
        # self.training = self.model.training

    def perturb(self, inputs, target):
        self.model.eval()
        _model_freeze(self.model)
        # return self.attack(x, target)

        batch_size = inputs.size(0)
        m = batch_size
        n = batch_size

        x = inputs.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        # logits_pred_nat, fea_nat = self.model(inputs)
        logits_pred_nat = self.model(inputs)

        for _ in range(self.iterations):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)

            # logits_pred, fea = self.model(x)
            logits_pred = self.model(x)

            ot_loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
                                                  logits_pred, None, None,
                                                  0.01, m, n)

            self.model.zero_grad()
            adv_loss = ot_loss
            adv_loss.backward(retain_graph=True)
            x_adv = x.data + self.step * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
            
        x = Variable(x_adv)
        return x
        # return logits_pred, adv_loss

