from tqdm.auto import tqdm
from utils import AverageMeter
import torch
from utils import collect
from attacker import LinfPGD

def _model_freeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=False

def _model_unfreeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=True

def ele_attack(adversary, inputs, target, device, frac=8):
    this_shape = target.shape[0]//frac
    noisy = torch.rand_like(inputs)
    noisy = noisy[:this_shape,:]
    eleven = target[:this_shape,:]
    
    adversary.targeted = True
    adv_inputs = adversary.perturb(noisy, target).detach()
    return adv_inputs, eleven


def untarget_attack(adversary, inputs, true_target):
    adversary.targeted = False
    return adversary.perturb(inputs, true_target).detach()

class EleRunner():
    def __init__(self, epochs, model, train_loader, test_loader, criterion, optimizer, scheduler, attacker, num_class, device):
        self.device = device
        self.epochs = epochs
        self.num_class = num_class
        self.eval_interval = 20

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.attacker = attacker
        self.std_attacker = LinfPGD(model, epsilon=8/255, step=2/255, iterations=20, random_start=True, targeted=False)

        self.desc = lambda status, progress: f"{status}: {progress}"
        
    def clean_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Clean train", progress))
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()

    def adv_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            _model_freeze(self.model)
            noisy, eleven = ele_attack(self.attacker, data, target, self.device)
            eleven *= self.num_class
            _model_unfreeze(self.model)

            inputs = torch.cat((data, noisy), dim=0)
            target = torch.cat((target, eleven), dim=0)

            output = self.model(inputs)
            loss = self.criterion(output, target)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()
    
    def train_eval(self, progress):
        self.model.eval()
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()
        with torch.no_grad():
            pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Train eval", progress))
            for batch_idx, (data, target) in enumerate(self.train_loader):
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

    def adv_eval(self, progress):
        self.model.eval()
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()
        
        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            _model_freeze(self.model)
            data = untarget_attack(self.attacker, data, target)
            _model_unfreeze(self.model)
            
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

    def train(self, adv=True):
        (avg_loss, acc_sum, acc_count) = self.adv_eval("Adv init")
        avg_loss = collect(avg_loss, self.device)
        avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            tqdm.write("Eval (Adver) init, Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

        (avg_loss, acc_sum, acc_count) = self.clean_eval("Clean init")
        avg_loss = collect(avg_loss, self.device)
        avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            tqdm.write("Eval (Clean) init, Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))
        

        for epoch_idx in range(self.epochs):
            if adv:
                avg_loss = self.adv_step("{}/{}".format(epoch_idx, self.epochs))
            else:
                avg_loss = self.clean_step("{}/{}".format(epoch_idx, self.epochs))
            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                if adv:
                    tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
                else:
                    tqdm.write("Clean training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            if self.scheduler is not None:
                self.scheduler.step()

            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                avg_loss, acc_sum, acc_count = self.train_eval("{}/{}".format(epoch_idx, self.epochs))
                avg_loss = collect(avg_loss, self.device)
                avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
                if torch.distributed.get_rank() == 0:
                    tqdm.write("Eval (Train) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))
                
                avg_loss, acc_sum, acc_count = self.clean_eval("{}/{}".format(epoch_idx, self.epochs))
                avg_loss = collect(avg_loss, self.device)
                avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
                if torch.distributed.get_rank() == 0:
                    tqdm.write("Eval (Clean) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))
                    
                avg_loss, acc_sum, acc_count = self.adv_eval("{}/{}".format(epoch_idx, self.epochs))
                avg_loss = collect(avg_loss, self.device)
                avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
                if torch.distributed.get_rank() == 0:
                    tqdm.write("Eval (Adver) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
    