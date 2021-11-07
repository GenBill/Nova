import torch
import torch.nn as nn

from tqdm.auto import tqdm

from utils import AverageMeter
from utils import collect

from advertorch.attacks import FGSM as atk_FGSM
from advertorch.attacks import LinfPGDAttack as atk_PGD
from advertorch.attacks import CarliniWagnerL2Attack as atk_CW

from attacker import my_APGDAttack_targeted
from autoattack.square import SquareAttack

def untarget_attack(adversary, inputs, true_target):
    adversary.targeted = False
    return adversary.perturb(inputs, true_target).detach()

class EvalRunner():
    def __init__(self, model, num_classes, test_loader, criterion, device):
        self.device = device

        self.model = model
        self.num_classes = num_classes
        self.test_loader = test_loader
        self.criterion = criterion

        self.desc = lambda status, progress: f"{status}: {progress}"

    def clean_eval(self, progress):
        self.model.eval()
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
        
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def FGSM_eval(self, progress):
        self.model.eval()
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

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
        
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def PGD_eval(self, progress, nb_iter=20):
        self.model.eval()
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        attacker = atk_PGD(
            self.model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=8/255, eps_iter=2/255, nb_iter=nb_iter, 
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
        
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def myPGD_eval(self, progress, nb_iter=20):
        self.model.eval()
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
        
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)

    def CW_eval(self, progress, search_steps=4, nb_iter=20):
        self.model.eval()
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        attacker = atk_CW(self.model, self.num_classes, binary_search_steps=search_steps, max_iterations=nb_iter)
        
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
        
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def Lipz_eval(self, nb_iter=100):
        self.model.eval()
        attacker = atk_PGD(
            self.model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=8/255, eps_iter=2/255, nb_iter=nb_iter, 
            rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
        )

        all_Lipz = 0
        sample_size = len(self.test_loader.dataset)

        for example_0, labels in tqdm(self.test_loader):
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

        return all_Lipz

    def Square_eval(self, progress, nb_iter=20):
        self.model.eval()
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        attacker = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
            n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
        
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
        
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)