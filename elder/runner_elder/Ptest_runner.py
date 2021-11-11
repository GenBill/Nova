from torch._C import Size
from tqdm.auto import tqdm
from utils import AverageMeter
from utils import collect

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# from advertorch.attacks import LinfPGDAttack

def soft_loss(pred, soft_targets):
    return torch.sqrt(nn.MSELoss()(pred, soft_targets))

# def soft_loss(pred, soft_targets):
#     logsoftmax = nn.LogSoftmax(dim=1)
#     return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), dim=1))

def std_target_attack(adversary, inputs, true_target, num_class, device):
    target = torch.randint(low=0, high=num_class-1, size=true_target.shape, device=device)
    # Ensure target != true_target
    target += (target >= true_target).int()
    adversary.targeted = True

    ret_inputs = adversary.perturb(inputs, target).detach()
    ret_target = F.one_hot(true_target, num_class).to(device)
 
    return ret_inputs, ret_target

def std_untarget_attack(adversary, inputs, true_target):
    adversary.targeted = False
    return adversary.perturb(inputs, true_target).detach()


class PtestRunner():
    def __init__(self, model, test_loader, criterion, attacker, num_class, device):
        self.device = device

        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.mse = nn.MSELoss(reduction='none')

        self.attacker = attacker
        self.num_class = num_class

    def tar_adv_eval(self):
        self.model.eval()

        total = 0
        linespace = 100
        loss_list = torch.zeros(linespace, device=self.device)
        acc_list = torch.zeros(linespace, device=self.device)
        lipz_list = torch.zeros(linespace, device=self.device)

        for batch_idx, (data, target) in enumerate(tqdm(self.test_loader)):
            total += data.shape[0]
            data, target = data.to(self.device), target.to(self.device)
            hotarget = F.one_hot(target, self.num_class).to(self.device)
            datadv_1, _ = std_target_attack(self.attacker, data, target, self.num_class, self.device)
            datadv_2, _ = std_target_attack(self.attacker, data, target, self.num_class, self.device)
            true_output = self.model(data)

            for index in range(linespace):
                
                with torch.no_grad():
                    lamb = float(index)/float(linespace)
                    datadv = lamb * datadv_1 + (1-lamb) *datadv_2

                    output = self.model(datadv)
                    loss = self.criterion(output, target)
                    loss_list[index] += loss.item()
                    
                    pred = output.argmax(dim=1)
                    true_positive = (pred == target).sum().item()
                    acc_list[index] += true_positive

                    diff = torch.sum(torch.abs(true_output-output).view(data.shape[0], -1), dim=1)
                    pleng = torch.max(torch.abs(data-datadv).view(data.shape[0], -1), dim=1)[0]
                    lipz_list[index] += torch.sum(diff/pleng)

        loss_list /= total
        acc_list /= total
        lipz_list /= total

        return loss_list, acc_list, lipz_list

    def untar_adv_eval(self):
        self.model.eval()

        total = 0
        linespace = 100
        loss_list = torch.zeros(linespace, device=self.device)
        acc_list = torch.zeros(linespace, device=self.device)
        lipz_list = torch.zeros(linespace, device=self.device)

        for batch_idx, (data, target) in enumerate(tqdm(self.test_loader)):
            total += data.shape[0]
            data, target = data.to(self.device), target.to(self.device)
            hotarget = F.one_hot(target, self.num_class).to(self.device)
            datadv_1 = std_untarget_attack(self.attacker, data, target)
            datadv_2 = std_untarget_attack(self.attacker, data, target)
            true_output = self.model(data)

            for index in range(linespace):
                
                with torch.no_grad():
                    lamb = float(index)/float(linespace)
                    datadv = lamb * datadv_1 + (1-lamb) *datadv_2

                    output = self.model(datadv)
                    loss = self.criterion(output, target)
                    loss_list[index] += loss.item()
                    
                    pred = output.argmax(dim=1)
                    true_positive = (pred == target).sum().item()
                    acc_list[index] += true_positive

                    diff = torch.sum(torch.abs(true_output-output).view(data.shape[0], -1), dim=1)
                    pleng = torch.max(torch.abs(data-datadv).view(data.shape[0], -1), dim=1)[0]
                    lipz_list[index] += torch.sum(diff/pleng)

        loss_list /= total
        acc_list /= total
        lipz_list /= total

        return loss_list, acc_list, lipz_list
    
    def onetar_adv_eval(self):
        self.model.eval()

        total = 0
        linespace = 200
        loss_list = torch.zeros(linespace, device=self.device)
        acc_list = torch.zeros(linespace, device=self.device)
        lipz_list = torch.zeros(linespace, device=self.device)

        for batch_idx, (data, target) in enumerate(tqdm(self.test_loader)):
            total += data.shape[0]
            data, target = data.to(self.device), target.to(self.device)
            hotarget = F.one_hot(target, self.num_class).to(self.device)
            datadv_1, _ = std_target_attack(self.attacker, data, target, self.num_class, self.device)
            true_output = self.model(data)

            for index in range(linespace):
                
                with torch.no_grad():
                    lamb = 2*float(index)/float(linespace)
                    datadv = lamb * datadv_1 + (1-lamb) *data

                    output = self.model(datadv)
                    loss = self.criterion(output, target)
                    loss_list[index] += loss.item()
                    
                    pred = output.argmax(dim=1)
                    true_positive = (pred == target).sum().item()
                    acc_list[index] += true_positive

                    diff = torch.sum(torch.abs(true_output-output).view(data.shape[0], -1), dim=1)
                    pleng = torch.max(torch.abs(data-datadv).view(data.shape[0], -1), dim=1)[0]
                    lipz_list[index] += torch.sum(diff/pleng)

        loss_list /= total
        acc_list /= total
        lipz_list /= total

        return loss_list, acc_list, lipz_list

    def oneuntar_adv_eval(self):
        self.model.eval()

        total = 0
        linespace = 200
        loss_list = torch.zeros(linespace, device=self.device)
        acc_list = torch.zeros(linespace, device=self.device)
        lipz_list = torch.zeros(linespace, device=self.device)

        for batch_idx, (data, target) in enumerate(tqdm(self.test_loader)):
            total += data.shape[0]
            data, target = data.to(self.device), target.to(self.device)
            hotarget = F.one_hot(target, self.num_class).to(self.device)
            datadv_1 = std_untarget_attack(self.attacker, data, target)
            true_output = self.model(data)

            for index in range(linespace):
                
                with torch.no_grad():
                    lamb = 2*float(index)/float(linespace)
                    datadv = lamb * datadv_1 + (1-lamb) *data

                    output = self.model(datadv)
                    loss = self.criterion(output, target)
                    loss_list[index] += loss.item()
                    
                    pred = output.argmax(dim=1)
                    true_positive = (pred == target).sum().item()
                    acc_list[index] += true_positive

                    diff = torch.sum(torch.abs(true_output-output).view(data.shape[0], -1), dim=1)
                    pleng = torch.max(torch.abs(data-datadv).view(data.shape[0], -1), dim=1)[0]
                    lipz_list[index] += torch.sum(diff/pleng)

        loss_list /= total
        acc_list /= total
        lipz_list /= total

        return loss_list, acc_list, lipz_list