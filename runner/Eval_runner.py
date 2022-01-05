import torch
import torch.nn as nn

from tqdm.auto import tqdm

from utils import AverageMeter
from utils import collect

from advertorch.attacks import FGSM as atk_FGSM
from advertorch.attacks import LinfPGDAttack as atk_PGD
from advertorch.attacks import CarliniWagnerL2Attack as atk_CW
from advertorch.attacks import LinfSPSAAttack

from attacker import my_APGDAttack_targeted, StarKnifePGD, RiemKnifePGD
from autoattack.square import SquareAttack

def _model_freeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=False

def _model_unfreeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=True

def untarget_attack(adversary, inputs, true_target):
    # adversary.targeted = False
    return adversary.perturb(inputs, true_target).detach()

import numpy as np
def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes, targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max((1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss

class EvalRunner():
    def __init__(self, model, num_classes, test_loader, criterion, device):
        self.device = device

        self.model = model
        self.num_classes = num_classes
        self.test_loader = test_loader
        self.criterion = criterion
        self.cwloss = CWLoss(num_classes, reduce=False)

        self.desc = lambda status, progress: f"{status}: {progress}"

    def clean_eval(self, progress):
        self.model.eval()
        _model_freeze(self.model)
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()
        with torch.no_grad():
            pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Clean eval", progress))
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                loss_meter.update(loss.item())
                pred = output.argmax(dim=1)

                true_positive = (pred == target).sum().item()
                total = pred.shape[0]
                accuracy_meter.update(true_positive, total)
                
            pbar.update(1)
        pbar.close()
        
        _model_unfreeze(self.model)
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def FGSM_eval(self, progress, reloss=False):
        self.model.eval()
        _model_freeze(self.model)
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        if reloss:
            attacker = atk_FGSM(self.model, self.criterion, eps=8/255)
        else:
            attacker = atk_FGSM(self.model, eps=8/255)
        
        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = untarget_attack(attacker, data, target)
            
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
                loss_meter.update(loss.item())
                pred = output.argmax(dim=1)

                true_positive = (pred == target).sum().item()
                total = pred.shape[0]
                accuracy_meter.update(true_positive, total)
            pbar.update(1)

        pbar.close()
        
        _model_unfreeze(self.model)
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def PGD_eval(self, progress, nb_iter=20, reloss=False):
        self.model.eval()
        _model_freeze(self.model)
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        if reloss:
            attacker = atk_PGD(
                self.model, loss_fn=self.criterion, eps=8/255, eps_iter=2/255, nb_iter=nb_iter, 
                rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
            )
        else:
            attacker = atk_PGD(
                self.model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255, eps_iter=2/255, nb_iter=nb_iter, 
                rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
            )
        
        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = untarget_attack(attacker, data, target)
            
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
                loss_meter.update(loss.item())
                pred = output.argmax(dim=1)

                true_positive = (pred == target).sum().item()
                total = pred.shape[0]
                accuracy_meter.update(true_positive, total)
            pbar.update(1)

        pbar.close()
        
        _model_unfreeze(self.model)
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def CWnum_eval(self, progress, nb_iter=20, reloss=False):
        self.model.eval()
        _model_freeze(self.model)
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        if reloss:
            attacker = atk_PGD(
                self.model, loss_fn=self.criterion, eps=8/255, eps_iter=2/255, nb_iter=nb_iter, 
                rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
            )
        else:
            attacker = atk_PGD(
                self.model, loss_fn=self.cwloss, eps=8/255, eps_iter=2/255, nb_iter=nb_iter, 
                rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
            )
        
        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = untarget_attack(attacker, data, target)
            
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
                loss_meter.update(loss.item())
                pred = output.argmax(dim=1)

                true_positive = (pred == target).sum().item()
                total = pred.shape[0]
                accuracy_meter.update(true_positive, total)
            pbar.update(1)

        pbar.close()
        
        _model_unfreeze(self.model)
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def StarKnife_eval(self, progress, nb_iter=100, mana=4, class_num=10, rand_init=False, targeted=False, doubled=True):
        self.model.eval()
        _model_freeze(self.model)
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        attacker = StarKnifePGD(
            self.model, eps=8/255, eps_iter=1/255, nb_iter=nb_iter, 
            mana=mana, class_num=class_num, rand_init=rand_init, targeted=targeted, doubled=doubled
        )
        
        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = untarget_attack(attacker, data, target)
            
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
                loss_meter.update(loss.item())
                pred = output.argmax(dim=1)

                true_positive = (pred == target).sum().item()
                total = pred.shape[0]
                accuracy_meter.update(true_positive, total)
            pbar.update(1)

        pbar.close()
        
        _model_unfreeze(self.model)
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def RiemKnife_eval(self, progress, nb_iter=100, mana=4, class_num=10, rand_init=False, targeted=False, doubled=True):
        self.model.eval()
        _model_freeze(self.model)
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        attacker = RiemKnifePGD(
            self.model, eps=8/255, eps_iter=2/255, nb_iter=nb_iter, 
            mana=mana, class_num=class_num, rand_init=rand_init, targeted=targeted, doubled=doubled
        )
        
        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = untarget_attack(attacker, data, target)
            
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
                loss_meter.update(loss.item())
                pred = output.argmax(dim=1)

                true_positive = (pred == target).sum().item()
                total = pred.shape[0]
                accuracy_meter.update(true_positive, total)
            pbar.update(1)

        pbar.close()
        
        _model_unfreeze(self.model)
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def myPGD_eval(self, progress, nb_iter=20):
        self.model.eval()
        _model_freeze(self.model)
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        attacker = my_APGDAttack_targeted(
            self.model, eps=8/255, n_iter=nb_iter, 
            n_target_classes=self.num_classes-1, device=self.device
        )
        
        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("myAdv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = untarget_attack(attacker, data, target)
            
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
                loss_meter.update(loss.item())
                pred = output.argmax(dim=1)

                true_positive = (pred == target).sum().item()
                total = pred.shape[0]
                accuracy_meter.update(true_positive, total)
            pbar.update(1)

        pbar.close()
        
        _model_unfreeze(self.model)
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)

    def CW_eval(self, progress, search_steps=1, nb_iter=100):
        self.model.eval()
        _model_freeze(self.model)
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        attacker = atk_CW(self.model, self.num_classes, binary_search_steps=search_steps, max_iterations=nb_iter, initial_const=1)
        
        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = untarget_attack(attacker, data, target)
            
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
                loss_meter.update(loss.item())
                pred = output.argmax(dim=1)

                true_positive = (pred == target).sum().item()
                total = pred.shape[0]
                accuracy_meter.update(true_positive, total)
            pbar.update(1)

        pbar.close()
        
        _model_unfreeze(self.model)
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def Lipz_eval(self, nb_iter=100, reloss=False):
        self.model.eval()
        _model_freeze(self.model)
        if reloss:
            attacker = atk_PGD(
                self.model, loss_fn=self.criterion, eps=8/255, eps_iter=2/255, nb_iter=nb_iter, 
                rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
            )
        else:
            attacker = atk_PGD(
                self.model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255, eps_iter=2/255, nb_iter=nb_iter, 
                rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
            )

        all_Lipz = 0
        sample_size = len(self.test_loader.dataset)

        for example_0, labels in self.test_loader:
            labels = labels.to(self.device)
            example_0 = example_0.to(self.device)
            example_1 = attacker.perturb(example_0, labels).detach()

            # do a forward pass on the example
            pred_0 = self.model(example_0)
            pred_1 = self.model(example_1)
            
            diff_pred = pred_1 - pred_0
            leng_diff_pred = torch.sum(torch.abs(diff_pred), dim=1)

            diff_input = example_1 - example_0
            leng_diff_input = torch.max(diff_input, dim=3).values
            leng_diff_input = torch.max(leng_diff_input, dim=2).values
            leng_diff_input = torch.max(leng_diff_input, dim=1).values
            
            # Count diff / rand_vector
            Ret = leng_diff_pred / leng_diff_input
            Local_Lipz = torch.sum(Ret, dim=0).item()
            
            all_Lipz += Local_Lipz / sample_size

        _model_unfreeze(self.model)
        return all_Lipz

    def Square_eval(self, progress, nb_iter=20):
        self.model.eval()
        _model_freeze(self.model)
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        attacker = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=8/255, norm='Linf',
            n_restarts=1, verbose=False, device=self.device, resc_schedule=False)
        
        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = untarget_attack(attacker, data, target)
            
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
                loss_meter.update(loss.item())
                pred = output.argmax(dim=1)

                true_positive = (pred == target).sum().item()
                total = pred.shape[0]
                accuracy_meter.update(true_positive, total)
            pbar.update(1)

        pbar.close()
        
        _model_unfreeze(self.model)
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def SPSA_eval(self, progress, nb_iter=20):
        self.model.eval()
        _model_freeze(self.model)
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        attacker = LinfSPSAAttack(self.model, eps=8/255, delta=0.01, lr=0.01, nb_iter=nb_iter,
            nb_sample=128, max_batch_size=64, targeted=False,
            loss_fn=None, clip_min=0.0, clip_max=1.0
            # loss_fn=nn.CrossEntropyLoss(reduction="none"), clip_min=0.0, clip_max=1.0
        )
        
        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = untarget_attack(attacker, data, target)
            
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
                loss_meter.update(loss.item())
                pred = output.argmax(dim=1)

                true_positive = (pred == target).sum().item()
                total = pred.shape[0]
                accuracy_meter.update(true_positive, total)
            pbar.update(1)

        pbar.close()
        
        _model_unfreeze(self.model)
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def SPSA_CE_eval(self, progress, nb_iter=20):
        self.model.eval()
        _model_freeze(self.model)
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        attacker = LinfSPSAAttack(self.model, eps=8/255, delta=0.01, lr=0.01, nb_iter=nb_iter,
            nb_sample=128, max_batch_size=64, targeted=False,
            loss_fn=nn.CrossEntropyLoss(reduction="none"), clip_min=0.0, clip_max=1.0
        )
        
        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = untarget_attack(attacker, data, target)
            
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
                loss_meter.update(loss.item())
                pred = output.argmax(dim=1)

                true_positive = (pred == target).sum().item()
                total = pred.shape[0]
                accuracy_meter.update(true_positive, total)
            pbar.update(1)

        pbar.close()
        
        _model_unfreeze(self.model)
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
