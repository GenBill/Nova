from torch._C import Size
from tqdm.auto import tqdm
from utils import AverageMeter
from utils import collect
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import copy

from attacker import LinfPGD
from advertorch.attacks import LinfPGDAttack

from runner.my_Rand import btower_Rand as tower_Rand
from runner.my_Rand import brain_Rand as rain_Rand

def soft_loss(pred, soft_targets):
    # return torch.mean(torch.sqrt(torch.mean((pred-soft_targets)**2, dim=1)), dim=0)
    return torch.mean(torch.norm(pred-soft_targets, dim=1), dim=0)

def untarget_attack(adversary, inputs, true_target):
    adversary.targeted = False
    return adversary.perturb(inputs, true_target).detach()

def target_attack(adversary, inputs, true_target, num_class, device):
    # Ensure target != true_target
    target = torch.randint(low=0, high=num_class-1, size=true_target.shape, device=device)
    target += (target >= true_target).int()
    
    adversary.targeted = True
    return adversary.perturb(inputs, target).detach()

def duelink_attack(adversary, inputs, true_target):
    adversary.targeted = True
    return adversary.perturb(inputs, true_target).detach()

def Vertex_Mix(adv_inputs, inputs, device):
    rand_lambda = tower_Rand((inputs.shape[0],1,1,1), device=device)
    return rand_lambda*adv_inputs + (1-rand_lambda)*inputs

def Edge_Mix(adv_inputs_1, adv_inputs_2, device):
    rand_lambda = rain_Rand((adv_inputs_1.shape[0],1,1,1), device=device)
    return rand_lambda*adv_inputs_1 + (1-rand_lambda)*adv_inputs_2

def Plain_Mix(inputs_1, inputs_2, device):
    rand_lambda = torch.rand((inputs_1.shape[0],1,1,1), device=device)
    return rand_lambda*inputs_1 + (1-rand_lambda)*inputs_2

class FrostRunner():
    def __init__(self, epochs, model, train_loader, test_loader, criterion, optimizer, scheduler, attacker, num_class, device):
        self.device = device
        self.epochs = epochs
        self.eval_interval = 20

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.attacker = attacker
        self.std_attacker = LinfPGD(model, epsilon=8/255, step=2/255, iterations=20, random_start=True, targeted=False)
        self.lipz_attacker = LinfPGD(model, epsilon=8/255, step=2/255, iterations=100, random_start=True, targeted=False)
        
        # self.std_attacker = LinfPGDAttack(
        #     self.model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=8/255, eps_iter=2/255, nb_iter=20, 
        #     rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, )
        # self.lipz_attacker = LinfPGDAttack(
        #     self.model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=8/255, eps_iter=2/255, nb_iter=100, 
        #     rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, )

        self.num_class = num_class

        self.desc = lambda status, progress: f"{status}: {progress}"
        
    def add_shower(self, epoch_idx):
        # train test
        avg_loss, acc_sum, acc_count = self.train_eval("{}/{}".format(epoch_idx, self.epochs))
        avg_loss = collect(avg_loss, self.device)
        avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            tqdm.write("Eval (Train) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))
        
        # clean test
        avg_loss, acc_sum, acc_count = self.clean_eval("{}/{}".format(epoch_idx, self.epochs))
        avg_loss = collect(avg_loss, self.device)
        avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            tqdm.write("Eval (Clean) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))

        # std adv test
        avg_loss, acc_sum, acc_count = self.std_adv_eval("{}/{}".format(epoch_idx, self.epochs))
        avg_loss = collect(avg_loss, self.device)
        avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            tqdm.write("Eval (Adver) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))

    def add_writer(self, writer, epoch_idx):
        # train test
        avg_loss, acc_sum, acc_count = self.train_eval("{}/{}".format(epoch_idx, self.epochs))
        avg_loss = collect(avg_loss, self.device)
        avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            tqdm.write("Eval (Train) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))
            # writer.add_scalar("train_Acc", avg_acc, epoch_idx)
            # writer.add_scalar("train_Loss", avg_loss, epoch_idx)
        
        # clean test
        avg_loss, acc_sum, acc_count = self.clean_eval("{}/{}".format(epoch_idx, self.epochs))
        avg_loss = collect(avg_loss, self.device)
        avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            tqdm.write("Eval (Clean) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))
            writer.add_scalar("clean_Acc", avg_acc, epoch_idx)
            writer.add_scalar("clean_Loss", avg_loss, epoch_idx)

        # # now adv test
        # avg_loss, acc_sum, acc_count = self.adv_eval("{}/{}".format(epoch_idx, self.epochs))
        # avg_loss = collect(avg_loss, self.device)
        # avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        # if torch.distributed.get_rank() == 0:
        #     tqdm.write("Eval (Adver) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))
        #     writer.add_scalar("adv_Acc", avg_acc, epoch_idx)
        #     writer.add_scalar("adv_Loss", avg_loss, epoch_idx)

        # std adv test
        avg_loss, acc_sum, acc_count = self.std_adv_eval("{}/{}".format(epoch_idx, self.epochs))
        avg_loss = collect(avg_loss, self.device)
        avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            tqdm.write("Eval (Adver) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))
            writer.add_scalar("adv_Acc", avg_acc, epoch_idx)
            writer.add_scalar("adv_Loss", avg_loss, epoch_idx)
        
        # Lipz test
        # std_lipz = self.std_lipz_eval()
        adv_lipz = self.adv_lipz_eval()
        if torch.distributed.get_rank() == 0:
            tqdm.write("Eval (Lipz) {}/{}, adv Lipz. {:.6f}".format(epoch_idx, self.epochs, adv_lipz))
            # writer.add_scalar("std_Lipz", std_lipz, epoch_idx)
            writer.add_scalar("adv_Lipz", adv_lipz, epoch_idx)

    def wo_tar_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            inputs = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            labels = F.one_hot(labels, self.num_class).float()
            
            outputs = self.model(inputs)
            loss = soft_loss(outputs, labels)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()
    
    def wo_untar_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            inputs = untarget_attack(self.attacker, inputs, labels)
            labels = F.one_hot(labels, self.num_class).float()
            
            outputs = self.model(inputs)
            loss = soft_loss(outputs, labels)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()
    
    def wo_dl_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            inputs = duelink_attack(self.attacker, inputs, labels)
            labels = F.one_hot(labels, self.num_class).float()
            
            outputs = self.model(inputs)
            loss = soft_loss(outputs, labels)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()

    def vertex_tar_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            adv_inputs = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            inputs = Plain_Mix(adv_inputs, inputs, self.device)
            labels = F.one_hot(labels, self.num_class).float()
            
            outputs = self.model(inputs)
            loss = soft_loss(outputs, labels)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()
    
    def vertex_untar_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            adv_inputs = untarget_attack(self.attacker, inputs, labels)
            inputs = Plain_Mix(adv_inputs, inputs, self.device)
            labels = F.one_hot(labels, self.num_class).float()
            
            outputs = self.model(inputs)
            loss = soft_loss(outputs, labels)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()

    def vertex_dl_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            adv_inputs = duelink_attack(self.attacker, inputs, labels)
            inputs = Plain_Mix(adv_inputs, inputs, self.device)
            labels = F.one_hot(labels, self.num_class).float()
            
            outputs = self.model(inputs)
            loss = soft_loss(outputs, labels)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()

    def edge_tar_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            adv_inputs_1 = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            adv_inputs_2 = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            inputs = Plain_Mix(adv_inputs_1, adv_inputs_2, self.device)
            labels = F.one_hot(labels, self.num_class).float()
            
            outputs = self.model(inputs)
            loss = soft_loss(outputs, labels)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()
    
    def edge_untar_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            adv_inputs_1 = untarget_attack(self.attacker, inputs, labels)
            adv_inputs_2 = untarget_attack(self.attacker, inputs, labels)
            inputs = Plain_Mix(adv_inputs_1, adv_inputs_2, self.device)
            labels = F.one_hot(labels, self.num_class).float()
            
            outputs = self.model(inputs)
            loss = soft_loss(outputs, labels)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()

    def edge_dl_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            adv_inputs_1 = duelink_attack(self.attacker, inputs, labels)
            adv_inputs_2 = duelink_attack(self.attacker, inputs, labels)
            inputs = Plain_Mix(adv_inputs_1, adv_inputs_2, self.device)
            labels = F.one_hot(labels, self.num_class).float()
            
            outputs = self.model(inputs)
            loss = soft_loss(outputs, labels)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()

    def double_tar_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            adv_inputs_1 = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            adv_inputs_2 = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            adv_inputs_1 = Plain_Mix(adv_inputs_1, inputs, self.device)
            adv_inputs_2 = Plain_Mix(adv_inputs_2, inputs, self.device)

            inputs = Plain_Mix(adv_inputs_1, adv_inputs_2, self.device)
            labels = F.one_hot(labels, self.num_class).float()
            
            outputs = self.model(inputs)
            loss = soft_loss(outputs, labels)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()
    
    def double_untar_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            adv_inputs_1 = untarget_attack(self.attacker, inputs, labels)
            adv_inputs_2 = untarget_attack(self.attacker, inputs, labels)
            adv_inputs_1 = Plain_Mix(adv_inputs_1, inputs, self.device)
            adv_inputs_2 = Plain_Mix(adv_inputs_2, inputs, self.device)

            inputs = Plain_Mix(adv_inputs_1, adv_inputs_2, self.device)
            labels = F.one_hot(labels, self.num_class).float()
            
            outputs = self.model(inputs)
            loss = soft_loss(outputs, labels)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()

    def double_dl_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            adv_inputs_1 = duelink_attack(self.attacker, inputs, labels)
            adv_inputs_2 = duelink_attack(self.attacker, inputs, labels)
            adv_inputs_1 = Plain_Mix(adv_inputs_1, inputs, self.device)
            adv_inputs_2 = Plain_Mix(adv_inputs_2, inputs, self.device)

            inputs = Plain_Mix(adv_inputs_1, adv_inputs_2, self.device)
            labels = F.one_hot(labels, self.num_class).float()
            
            outputs = self.model(inputs)
            loss = soft_loss(outputs, labels)
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
            data = untarget_attack(self.attacker, data, target)
            
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
    
    def std_adv_eval(self, progress):
        self.model.eval()
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = untarget_attack(self.std_attacker, data, target)
            
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

    def adv_lipz_eval(self):
        self.model.eval()
        all_Lipz = 0
        sample_size = len(self.test_loader.dataset)

        for example_0, labels in self.test_loader:
            labels = labels.to(self.device)
            example_0 = example_0.to(self.device)
            example_1 = untarget_attack(self.lipz_attacker, example_0, labels)

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
    
    def wo_tar(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.wo_tar_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
    
    def wo_untar(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.wo_untar_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
    
    def wo_dl(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.wo_dl_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
    
    def vertex_tar(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.vertex_tar_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
    
    def vertex_untar(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.vertex_untar_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
    
    def vertex_dl(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.vertex_dl_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
    
    def edge_tar(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.edge_tar_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
    
    def edge_untar(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.edge_untar_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
    
    def edge_dl(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.edge_dl_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))

    def double_tar(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.double_tar_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
    
    def double_tar_writer(self, writer):
        ## Add a Writer
        self.add_writer(writer, 0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.double_tar_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_writer(writer, epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
    
    def double_untar(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.double_untar_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
    
    def double_dl(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.double_dl_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))

    def double_dl_writer(self, writer):
        ## Add a Writer
        self.add_writer(writer, 0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.double_dl_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_writer(writer, epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))