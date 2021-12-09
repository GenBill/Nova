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


def Image_Loss(x, y):
    batch_size = x.shape[0]
    # return torch.mean(torch.norm((x-y).view(batch_size,-1), dim=1))
    temp = torch.abs(x-y).view(batch_size,-1)
    # return torch.mean(torch.sqrt(torch.mean(temp*temp, dim=1)))
    return torch.mean(torch.mean(temp, dim=1))

'''
def Image_Loss(x, y):
    batch_size = x.shape[0]
    # return torch.mean(torch.norm((x-y).view(batch_size,-1), dim=1))
    temp = (torch.sign(x)*(y-x)**2).view(batch_size,-1)
    # return torch.mean(torch.sqrt(torch.mean(temp*temp, dim=1)))
    return torch.mean(torch.mean(temp, dim=1))
'''

class layertrans(nn.Module):
    def __init__(self, model):
        super(layertrans, self).__init__()
        self.model = model.module
        self.head = self.model.encoder[:4]
        self.block_0 = self.model.encoder[4]
        self.block_1 = self.model.encoder[5]
        self.block_2 = self.model.encoder[6]
        self.block_3 = self.model.encoder[7]
        self.tail = nn.Sequential(self.model.encoder[-2:], self.model.classifier)

    def forward(self, x):
        f0 = self.head(x)
        f1 = self.block_0(f0)
        f2 = self.block_1(f1)
        f3 = self.block_2(f2)
        f4 = self.block_3(f3)
        y = self.tail(f4)
        return f0, f1, f2, f3, f4, y

class RiemKnifePGD(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """
    def __init__(self, model, eps=8/255, eps_iter=2/255, nb_iter=20, mana=100, class_num=10, criterion=None, rand_init=True, targeted=False, doubled=True):
        super(RiemKnifePGD, self).__init__()
        # Arguments of PGD
        self.device = next(model.parameters()).device

        self.model = model
        self.trans_model = layertrans(model)
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

    def attack_single_run(self, data, target, adv_elder, mana_ind):
        if self.rand_init:
            x_adv = data.detach() + (torch.rand_like(data)*2-1)*self.eps
        else:
            # x_adv = data.detach()
            x_adv = Star_Init(self, data, adv_elder)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # steps_2 = self.nb_iter//4
        # steps_1 = self.nb_iter - steps_2

        for _ in range(self.nb_iter):
            x_adv.requires_grad_()
            with torch.no_grad():
                f0, f1, f2, f3, f4, _ = self.trans_model(data)
            f0_adv, f1_adv, f2_adv, f3_adv, f4_adv, output = self.trans_model(x_adv)
            self.trans_model.zero_grad()

            with torch.enable_grad():
                loss_adv = nn.functional.cross_entropy(output, target)
                loss_fea = Image_Loss(f0, f0_adv) + Image_Loss(f1, f1_adv) + Image_Loss(f2, f2_adv) + Image_Loss(f3, f3_adv) + Image_Loss(f4, f4_adv)
                # loss_star = 0
                # for n, i in enumerate(adv_elder):
                #     loss_star += Star_Loss(x_adv, i, data)/len(adv_elder)
                loss = loss_adv/1024 - loss_fea/(2**mana_ind)   #  - loss_star/10
            
            loss.backward()
            eta = self.eps_iter * x_adv.grad.sign()
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, data - self.eps), data + self.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # return x_adv
        # Finetune
        x_final = x_adv.detach() + (torch.rand_like(data)*2-1)*self.eps/8
        x_final = torch.clamp(x_final, 0.0, 1.0)
        for _ in range(20):
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
    
    def target_single_run(self, data, target, adv_elder, mana_ind):
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
            with torch.no_grad():
                f0, f1, f2, f3, f4, _ = self.trans_model(data)
            f0_adv, f1_adv, f2_adv, f3_adv, f4_adv, output = self.trans_model(x_adv)
            self.trans_model.zero_grad()

            with torch.enable_grad():
                loss_adv = nn.functional.cross_entropy(output, target)
                loss_fea = Image_Loss(f0, f0_adv) + Image_Loss(f1, f1_adv) + Image_Loss(f2, f2_adv) + Image_Loss(f3, f3_adv) + Image_Loss(f4, f4_adv)
                # loss_star = 0
                # for n, i in enumerate(adv_elder):
                #     loss_star += Star_Loss(x_adv, i, data)/len(adv_elder)
                loss = loss_adv/1024 - loss_fea/(4**mana_ind)   #  - loss_star/10
            
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

                    adv_curr = self.attack_single_run(x_to_fool, y_to_fool, adv_elder, mana_ind)
                    acc_curr = self.model(adv_curr).max(1)[1] == y_to_fool
                    ind_curr = (acc_curr == 0).nonzero().squeeze()
                    true_ind_curr = (acc_curr == 1).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    for i, adv_nth in enumerate(adv_elder):
                        adv_elder[i] = adv_nth[true_ind_curr].clone()
                    adv_elder.append(adv_curr[true_ind_curr].detach().clone())
                    ind_to_fool = ind_to_fool[true_ind_curr].clone()
                print(mana_ind, 'untar ACC', true_ind_curr.shape[0], '/', x.shape[0], '=', true_ind_curr.shape[0]/x.shape[0])
            
            for this_tar in range(self.class_num):
                adv_elder = []
                for mana_ind in range(self.mana):
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool = x[ind_to_fool].clone()
                        y_to_fool = y[ind_to_fool].clone()
                        tar_to_fool = this_tar*torch.ones_like(y_to_fool)

                        adv_curr = self.target_single_run(x_to_fool, tar_to_fool, adv_elder, mana_ind)
                        acc_curr = self.model(adv_curr).max(1)[1] == y_to_fool
                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        true_ind_curr = (acc_curr == 1).nonzero().squeeze()

                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        for i, adv_nth in enumerate(adv_elder):
                            adv_elder[i] = adv_nth[true_ind_curr].clone()
                        adv_elder.append(adv_curr[true_ind_curr].detach().clone())
                        ind_to_fool = ind_to_fool[true_ind_curr].clone()
                    print(this_tar, mana_ind, 'tar ACC', true_ind_curr.shape[0], '/', x.shape[0], '=', true_ind_curr.shape[0]/x.shape[0])

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

                        adv_curr = self.target_single_run(x_to_fool, tar_to_fool, adv_elder, mana_ind)
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

                    adv_curr = self.attack_single_run(x_to_fool, y_to_fool, adv_elder, mana_ind)
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
