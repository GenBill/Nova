import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
# import ot

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
        # self.model.eval()
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

            ot_loss = sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
                                                  logits_pred, None, None,
                                                  0.01, m, n, self.device)

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


## OT Part

def sinkhorn_loss_joint_IPOT(alpha, beta, x_feature, y_feature, x_label,
                             y_label, epsilon, m, n, device):

    C_fea = get_cost_matrix(x_feature, y_feature)
    C = C_fea
    T = sinkhorn(C, 0.01, 100, device)
    # T = IPOT(C, 1)
    batch_size = C.size(0)
    cost_ot = torch.sum(T * C)
    return cost_ot


def sinkhorn(C, epsilon, niter=50, device='cuda'):
    m = C.size(0)
    n = C.size(1)
    mu = Variable(1. / m * torch.FloatTensor(m).fill_(1).to(device),
                  requires_grad=False)
    nu = Variable(1. / n * torch.FloatTensor(n).fill_(1).to(device),
                  requires_grad=False)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) +
                         1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).cpu().data.numpy():
            break
    U, V = u, v

    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    pi = pi.to(device).float()
    return pi  # return the transport


def get_cost_matrix(x_feature, y_feature):
    C_fea = cost_matrix_cos(x_feature, y_feature)  # Wasserstein cost function
    return C_fea


def cost_matrix_cos(x, y, p=2):
    # return the m*n sized cost matrix
    "Returns the matrix of $|x_i-y_j|^p$."
    # un squeeze differently so that the tensors can broadcast
    # dim-2 (summed over) is the feature dim
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)

    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    c = torch.clamp(1 - cos(x_col, y_lin), min=0)

    return c
