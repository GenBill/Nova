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

class StarKnifePGD(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """
    def __init__(self, model, eps=8/255, eps_iter=2/255, nb_iter=20, mana=100, criterion=None, rand_init=True, targeted=False):
        super(StarKnifePGD, self).__init__()
        # Arguments of PGD
        self.device = next(model.parameters()).device

        self.model = model
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.mana = mana
        self.rand_init = rand_init
        self.targeted = targeted    # False

        self.criterion = criterion
        if self.criterion is None:
            self.criterion = lambda model, input, target: nn.functional.cross_entropy(model(input), target)

    def attack_single_run(self, data, target, adv_elder, fool_size):
        if self.rand_init:
            x_adv = data.detach() + (torch.rand_like(data)*2-1)*self.eps
        else:
            x_adv = data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.nb_iter):
            x_adv.requires_grad_()
            output = self.model(x_adv)
            self.model.zero_grad()
            with torch.enable_grad():
                loss_adv = nn.functional.cross_entropy(output, target)
                loss_star = 0
                for i in adv_elder:
                    loss_star += torch.mean(torch.cosine_similarity((x_adv-data).view(fool_size,-1), (i-data).view(fool_size,-1), dim=1))
                if len(adv_elder):
                    loss = loss_adv - loss_star/len(adv_elder)
                else:
                    loss = loss_adv
            loss.backward()
            eta = self.eps_iter * x_adv.grad.sign()
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, data - self.eps), data + self.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        return x_adv


    def perturb(self, x, y):
        ## Init Mana
        self.model.eval()
        _model_freeze(self.model)
        
        ## First Test
        x = x.detach().clone().float().to(self.device)
        y_pred = self.model(x).max(1)[1]
        
        acc = y_pred == y
        ind_to_fool = acc.nonzero().squeeze()
        adv = x.clone()
        adv_elder = []

        for mana_ind in range(self.mana):
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()

                adv_curr = self.attack_single_run(x_to_fool, y_to_fool, adv_elder, ind_to_fool.shape[0])
                acc_curr = self.model(adv_curr).max(1)[1] == y_to_fool
                ind_curr = (acc_curr == 0).nonzero().squeeze()
                true_ind_curr = (acc_curr == 1).nonzero().squeeze()

                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                for i, adv_nth in enumerate(adv_elder):
                    adv_elder[i] = adv_nth[true_ind_curr].clone()
                adv_elder.append(adv_curr[true_ind_curr].detach().clone())
                ind_to_fool = ind_to_fool[true_ind_curr].clone()
                # print('this ACC', true_ind_curr.shape[0], '/', x.shape[0], '=', true_ind_curr.shape[0]/x.shape[0])
        
        print('ACC', true_ind_curr.shape[0], '/', x.shape[0], '=', true_ind_curr.shape[0]/x.shape[0])
        _model_unfreeze(self.model)
        return adv
