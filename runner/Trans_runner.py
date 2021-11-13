import torch
import torch.nn as nn

from tqdm.auto import tqdm

from utils import AverageMeter
from utils import collect

from advertorch.attacks import LinfPGDAttack as atk_PGD

def _model_freeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=False

def _model_unfreeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=True

def untarget_attack(adversary, inputs, true_target):
    adversary.targeted = False
    return adversary.perturb(inputs, true_target).detach()

class TransRunner():
    def __init__(self, model_0, model_1, num_classes, test_loader, criterion, device):
        self.device = device

        self.model_0 = model_0
        self.model_1 = model_1
        self.num_classes = num_classes
        self.test_loader = test_loader
        self.criterion = criterion

        self.desc = lambda status, progress: f"{status}: {progress}"
    
    def Trans_PGD(self, progress, nb_iter=20):
        self.model_0.eval()
        self.model_1.eval()
        _model_freeze(self.model)
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        attacker = atk_PGD(
            self.model_1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255, eps_iter=20/255, nb_iter=nb_iter, 
            rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, 
        )
        
        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = untarget_attack(attacker, data, target)
            
            with torch.no_grad():
                output = self.model_0(data)
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
    