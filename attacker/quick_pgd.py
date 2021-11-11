import torch
import torch.nn as nn
import numpy as np

class QLinfPGD(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """
    def __init__(self, model, epsilon=8/255, step=2/255, iterations=20, criterion=None, random_start=True, targeted=False):
        super(QLinfPGD, self).__init__()
        # Arguments of PGD
        self.device = next(model.parameters()).device

        self.model = model
        self.epsilon = epsilon
        self.step = step
        self.iterations = iterations
        self.rand_init = random_start
        self.targeted = targeted

        self.criterion = criterion
        if self.criterion is None:
            self.criterion = lambda model, input, target: nn.functional.cross_entropy(model(input), target)

        # Model status
        self.training = self.model.training

    def perturb(self, data, target):
        self.model.eval()
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, data.shape)).float().cuda() if self.rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        if self.targeted:
            for _ in range(self.iterations):
                x_adv.requires_grad_()
                output = self.model(x_adv)
                self.model.zero_grad()
                with torch.enable_grad():
                    loss_adv = nn.CrossEntropyLoss(reduction="sum")(output, target)
                loss_adv.backward()
                eta = self.step * x_adv.grad.sign()
                x_adv = x_adv.detach() - eta
                x_adv = torch.min(torch.max(x_adv, data - self.epsilon), data + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        else:
            for _ in range(self.iterations):
                x_adv.requires_grad_()
                output = self.model(x_adv)
                self.model.zero_grad()
                with torch.enable_grad():
                    loss_adv = nn.CrossEntropyLoss(reduction="sum")(output, target)
                loss_adv.backward()
                eta = self.step * x_adv.grad.sign()
                x_adv = x_adv.detach() + eta
                x_adv = torch.min(torch.max(x_adv, data - self.epsilon), data + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv
