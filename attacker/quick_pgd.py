import torch
import torch.nn as nn
import numpy as np

def _model_freeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=False

def _model_unfreeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=True

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
        _model_freeze(self.model)

        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, data.shape)).float().cuda() if self.rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        if self.targeted:
            for _ in range(self.iterations):
                x_adv.requires_grad_()
                output = self.model(x_adv)
                self.model.zero_grad()
                with torch.enable_grad():
                    loss_adv = nn.functional.cross_entropy(output, target)
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
                    loss_adv = nn.functional.cross_entropy(output, target)
                loss_adv.backward()
                eta = self.step * x_adv.grad.sign()
                x_adv = x_adv.detach() + eta
                x_adv = torch.min(torch.max(x_adv, data - self.epsilon), data + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

        _model_unfreeze(self.model)
        return x_adv



class SpaceLinfPGD(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """
    def __init__(self, model, epsilon=8/255, step=2/255, iterations=20, num_classes=10, criterion=None, random_start=True, targeted=False):
        super(SpaceLinfPGD, self).__init__()
        # Arguments of PGD
        self.device = next(model.parameters()).device

        self.model = model
        self.epsilon = epsilon
        self.step = step
        self.iterations = iterations
        self.rand_init = random_start
        self.targeted = targeted
        self.num_classes = num_classes

        self.criterion = criterion
        if self.criterion is None:
            self.criterion = lambda model, input, target: nn.functional.cross_entropy(model(input), target)

        # Model status
        self.training = self.model.training

        self.gamma = 0.1
        self.adv_list = []

    def untar_step(self, data, target):

        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, data.shape)).float().cuda() if self.rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.iterations):
            x_adv.requires_grad_()
            output = self.model(x_adv)
            self.model.zero_grad()
            with torch.enable_grad():
                loss_adv = nn.functional.cross_entropy(output, target)
                for adv in self.adv_list:
                    loss_adv += self.gamma*(...)
            loss_adv.backward()
            eta = self.step * x_adv.grad.sign()
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, data - self.epsilon), data + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    def target_step(self, data, faker):

        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, data.shape)).float().cuda() if self.rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.iterations):
            x_adv.requires_grad_()
            output = self.model(x_adv)
            self.model.zero_grad()
            with torch.enable_grad():
                loss_adv = nn.functional.cross_entropy(output, faker)
                for adv in self.adv_list:
                    loss_adv += self.gamma*(...)
            loss_adv.backward()
            eta = self.step * x_adv.grad.sign()
            x_adv = x_adv.detach() - eta
            x_adv = torch.min(torch.max(x_adv, data - self.epsilon), data + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    def perturb(self, data, target):
        self.model.eval()
        _model_freeze(self.model)

        for faker in range(self.num_classes):
            if faker==target:
                this_adv = 
        
        for faker in range(self.num_classes):   # target_class in range(2, self.n_target_classes + 2):
            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    
                    if not self.is_tf_model:
                        output = self.model(x_to_fool)
                    else:
                        output = self.model.predict(x_to_fool)
                    self.y_target = output.sort(dim=1)[1][:, -target_class]

                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('target class {}'.format(target_class),
                            '- restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

            



        _model_unfreeze(self.model)
        return x_adv