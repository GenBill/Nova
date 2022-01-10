from random import random
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import numpy as np

def _model_freeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=False

def _model_unfreeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=True

def Star_Loss(x_adv, i, data):
    fool_size = data.shape[0]
    return torch.mean(torch.abs(torch.cosine_similarity((x_adv-data).view(fool_size,-1), (i-data).view(fool_size,-1), dim=1)))
    return -torch.mean(torch.abs(x_adv-i))/10

def Star_Init(self, data, adv_elder):
    x_adv = data.detach() + (torch.rand_like(data)*2-1)*self.eps
    if len(adv_elder):
        for _ in range(100):
            x_adv = x_adv.requires_grad_()
            loss = 0
            for i in adv_elder:
                loss += Star_Loss(x_adv, i, data)
            # loss /= len(adv_elder)
            loss.backward()
            eta = self.eps_iter/2 * x_adv.grad.sign()
            x_adv = x_adv.detach() - eta
            x_adv = torch.min(torch.max(x_adv, data - self.eps), data + self.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

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
    def __init__(self, model, eps=8/255, eps_iter=2/255, nb_iter=20, mana=100, class_num=10, criterion=None, rand_init=True, targeted=False, doubled=True):
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
        self.doubled = doubled
        self.class_num = class_num

        self.criterion = criterion
        if self.criterion is None:
            self.criterion = lambda model, input, target: nn.functional.cross_entropy(model(input), target)

    def attack_single_run(self, data, target, adv_elder):
        if self.rand_init:
            x_adv = data.detach() + (torch.rand_like(data)*2-1)*self.eps
        else:
            # x_adv = data.detach()
            x_adv = Star_Init(self, data, adv_elder)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        steps_2 = self.nb_iter//4
        steps_1 = self.nb_iter - steps_2

        for _ in range(steps_1):
            x_adv.requires_grad_()
            output = self.model(x_adv)
            self.model.zero_grad()
            with torch.enable_grad():
                loss_adv = nn.functional.cross_entropy(output, target)
                loss_star = 0
                for n, i in enumerate(adv_elder):
                    loss_star += Star_Loss(x_adv, i, data)*(0.6**n)*1e-1
                loss = loss_adv - loss_star
            
            loss.backward()
            eta = self.eps_iter * x_adv.grad.sign()
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, data - self.eps), data + self.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # return x_adv
        # Finetune
        x_final = x_adv.detach() + (torch.rand_like(data)*2-1)*self.eps/8
        x_final = torch.clamp(x_final, 0.0, 1.0)
        for _ in range(steps_2):
            x_final.requires_grad_()
            output = self.model(x_final)
            self.model.zero_grad()
            with torch.enable_grad():
                loss = nn.functional.cross_entropy(output, target)
            
            loss.backward()
            eta = self.eps_iter * x_final.grad.sign()
            x_final = x_final.detach() + eta/8
            x_final = torch.min(torch.max(x_final, x_adv - self.eps/8), x_adv + self.eps/8)
            x_final = torch.min(torch.max(x_final, data - self.eps), data + self.eps)
            x_final = torch.clamp(x_final, 0.0, 1.0)
        
        x_adv = x_final.detach()
        return x_adv
    
    def target_single_run(self, data, target, adv_elder):
        if self.rand_init:
            x_adv = data.detach() + (torch.rand_like(data)*2-1)*self.eps
        else:
            # x_adv = data.detach()
            x_adv = Star_Init(self, data, adv_elder)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        steps_2 = self.nb_iter//4
        steps_1 = self.nb_iter - steps_2

        for _ in range(steps_1):
            x_adv.requires_grad_()
            output = self.model(x_adv)
            self.model.zero_grad()
            with torch.enable_grad():
                loss_adv = nn.functional.cross_entropy(output, target)
                loss_star = 0
                for n, i in enumerate(adv_elder):
                    loss_star += Star_Loss(x_adv, i, data)*(0.6**n)*1e-1
                loss = loss_adv + loss_star
            
            loss.backward()
            eta = self.eps_iter * x_adv.grad.sign()
            x_adv = x_adv.detach() - eta
            x_adv = torch.min(torch.max(x_adv, data - self.eps), data + self.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # Finetune 1
        x_final = x_adv.detach() + (torch.rand_like(data)*2-1)*self.eps/8
        x_final = torch.clamp(x_final, 0.0, 1.0)
        for _ in range(steps_2):
            x_final.requires_grad_()
            output = self.model(x_final)
            self.model.zero_grad()
            with torch.enable_grad():
                loss = nn.functional.cross_entropy(output, target)
            
            loss.backward()
            eta = self.eps_iter * x_final.grad.sign()
            x_final = x_final.detach() - eta/8
            x_final = torch.min(torch.max(x_final, x_adv - self.eps/8), x_adv + self.eps/8)
            x_final = torch.min(torch.max(x_final, data - self.eps), data + self.eps)
            x_final = torch.clamp(x_final, 0.0, 1.0)
        
        x_adv = x_final.detach()
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

        if self.doubled:
            for mana_ind in range(self.mana):
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()

                    adv_curr = self.attack_single_run(x_to_fool, y_to_fool, adv_elder)
                    acc_curr = self.model(adv_curr).max(1)[1] == y_to_fool
                    ind_curr = (acc_curr == 0).nonzero().squeeze()
                    true_ind_curr = (acc_curr == 1).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    for i, adv_nth in enumerate(adv_elder):
                        adv_elder[i] = adv_nth[true_ind_curr].clone()
                    adv_elder.append(adv_curr[true_ind_curr].detach().clone())
                    ind_to_fool = ind_to_fool[true_ind_curr].clone()
            
            for this_tar in range(self.class_num):
                adv_elder = []
                for mana_ind in range(self.mana):
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool = x[ind_to_fool].clone()
                        y_to_fool = y[ind_to_fool].clone()
                        tar_to_fool = this_tar*torch.ones_like(y_to_fool)

                        adv_curr = self.target_single_run(x_to_fool, tar_to_fool, adv_elder)
                        acc_curr = self.model(adv_curr).max(1)[1] == y_to_fool
                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        true_ind_curr = (acc_curr == 1).nonzero().squeeze()

                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        for i, adv_nth in enumerate(adv_elder):
                            adv_elder[i] = adv_nth[true_ind_curr].clone()
                        adv_elder.append(adv_curr[true_ind_curr].detach().clone())
                        ind_to_fool = ind_to_fool[true_ind_curr].clone()

            _model_unfreeze(self.model)
            return adv

        if self.targeted:
            for this_tar in range(self.class_num):
                adv_elder = []
                for mana_ind in range(self.mana):
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool = x[ind_to_fool].clone()
                        y_to_fool = y[ind_to_fool].clone()
                        tar_to_fool = this_tar*torch.ones_like(y_to_fool)

                        adv_curr = self.target_single_run(x_to_fool, tar_to_fool, adv_elder)
                        acc_curr = self.model(adv_curr).max(1)[1] == y_to_fool
                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        true_ind_curr = (acc_curr == 1).nonzero().squeeze()

                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        for i, adv_nth in enumerate(adv_elder):
                            adv_elder[i] = adv_nth[true_ind_curr].clone()
                        adv_elder.append(adv_curr[true_ind_curr].detach().clone())
                        ind_to_fool = ind_to_fool[true_ind_curr].clone()
                    # print(this_tar, mana_ind, 'tar ACC', true_ind_curr.shape[0], '/', x.shape[0], '=', true_ind_curr.shape[0]/x.shape[0])
            
            # print('ACC', true_ind_curr.shape[0], '/', x.shape[0], '=', true_ind_curr.shape[0]/x.shape[0])
            _model_unfreeze(self.model)
            return adv
        
        else:
            for mana_ind in range(self.mana):
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()

                    adv_curr = self.attack_single_run(x_to_fool, y_to_fool, adv_elder)
                    acc_curr = self.model(adv_curr).max(1)[1] == y_to_fool
                    ind_curr = (acc_curr == 0).nonzero().squeeze()
                    true_ind_curr = (acc_curr == 1).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    for i, adv_nth in enumerate(adv_elder):
                        adv_elder[i] = adv_nth[true_ind_curr].clone()
                    adv_elder.append(adv_curr[true_ind_curr].detach().clone())
                    ind_to_fool = ind_to_fool[true_ind_curr].clone()
                print(mana_ind, 'this ACC', true_ind_curr.shape[0], '/', x.shape[0], '=', true_ind_curr.shape[0]/x.shape[0])
            
            print('ACC', true_ind_curr.shape[0], '/', x.shape[0], '=', true_ind_curr.shape[0]/x.shape[0])
            _model_unfreeze(self.model)
            return adv


class DoubleKnifePGD(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """
    def __init__(self, model, mana=2, eps=8/255, eps_iter=2/255, nb_iter=20, class_num=10, criterion=None, rand_init=True, targeted=False):
        super(DoubleKnifePGD, self).__init__()
        # Arguments of PGD
        self.device = next(model.parameters()).device

        self.model = model
        self.mana = mana
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.rand_init = rand_init
        self.targeted = targeted    # False
        self.class_num = class_num

        self.criterion = criterion
        if self.criterion is None:
            self.criterion = lambda model, input, target: nn.functional.cross_entropy(model(input), target)

    def attack_single_run(self, data, target):
        x_adv = data.detach() + (torch.rand_like(data)*2-1)*self.eps
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.nb_iter):
            x_adv.requires_grad_()
            output = self.model(x_adv)
            self.model.zero_grad()
            with torch.enable_grad():
                loss = nn.functional.cross_entropy(output, target)
            
            loss.backward()
            eta = self.eps_iter * x_adv.grad.sign()
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, data - self.eps), data + self.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        return x_adv

    
    def target_single_run(self, data, target):
        x_adv = data.detach() + (torch.rand_like(data)*2-1)*self.eps
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.nb_iter):
            x_adv.requires_grad_()
            output = self.model(x_adv)
            self.model.zero_grad()
            with torch.enable_grad():
                loss = nn.functional.cross_entropy(output, target)
            
            loss.backward()
            eta = self.eps_iter * x_adv.grad.sign()
            x_adv = x_adv.detach() - eta
            x_adv = torch.min(torch.max(x_adv, data - self.eps), data + self.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        return x_adv


    def perturb(self, x, y):
        batch_size = y.shape[0]

        ## Init Mana
        self.model.eval()
        _model_freeze(self.model)
        
        ## First Test
        x = x.detach().clone().float().to(self.device)
        y_pred = self.model(x).max(1)[1]
        
        acc = y_pred == y
        ind_to_fool = acc.nonzero().squeeze()
        adv = x.clone()

        fake_target = torch.zeros(size=(batch_size, self.class_num), dtype=int, device=y.device)
        for i in range(batch_size):
            fake_target[i,:] = torch.randperm(self.class_num)

        for this_round in range(self.mana):
            this_fake_target = fake_target[:, this_round]

            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()
                tar_to_fool = this_fake_target[ind_to_fool].clone()

                adv_curr = self.target_single_run(x_to_fool, tar_to_fool)
                acc_curr = self.model(adv_curr).max(1)[1] == y_to_fool
                ind_curr = (acc_curr == 0).nonzero().squeeze()
                true_ind_curr = (acc_curr == 1).nonzero().squeeze()

                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

                ind_to_fool = ind_to_fool[true_ind_curr].clone()

        if len(ind_to_fool.shape) == 0:
            ind_to_fool = ind_to_fool.unsqueeze(0)
        if ind_to_fool.numel() != 0:
            x_to_fool = x[ind_to_fool].clone()
            y_to_fool = y[ind_to_fool].clone()

            adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
            adv[ind_to_fool] = adv_curr.clone()

        _model_unfreeze(self.model)
        return adv

