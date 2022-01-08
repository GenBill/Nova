from torch._C import Size
from tqdm.auto import tqdm
from utils import AverageMeter
from utils import collect as collect
from utils import MMC_Loss
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

def _model_freeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=False

def _model_unfreeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=True

def label_smoothing(onehot, n_classes, factor):
    return onehot * factor + (onehot - 1) * ((factor - 1)/(n_classes - 1))

## L2 Loss
def soft_loss(pred, soft_targets):
    # return torch.mean(torch.sqrt(torch.mean((pred-soft_targets)**2, dim=1)), dim=0)
    return torch.mean(torch.norm(pred-soft_targets, dim=1), dim=0)

## Wot Loss
# def soft_loss(pred, soft_targets):
#     # return torch.mean(torch.sqrt(torch.mean((pred-soft_targets)**2, dim=1)), dim=0)
#     return torch.norm(pred-soft_targets)

# def soft_loss(pred, soft_targets):
#     # logsoftmax = nn.LogSoftmax(dim=1)
#     return torch.mean(torch.sum(-soft_targets * F.log_softmax(pred, dim=1), dim=1))

def untarget_attack(adversary, inputs, true_target):
    adversary.targeted = False
    return adversary.perturb(inputs, true_target).detach()

def target_attack(adversary, inputs, true_target, num_class, device):
    # Ensure target != true_target
    target = torch.randint(low=0, high=num_class-1, size=true_target.shape, device=device)
    target += (target >= true_target).int()
    
    adversary.targeted = True
    return adversary.perturb(inputs, target).detach()

def top_attack(adversary, inputs, true_target, indexes, device, top=10):
    # Ensure target != true_target
    
    target_id = torch.randint(low=0, high=top-1, size=(true_target.shape[0], 1), device=device)
    target = torch.gather(indexes, dim=0, index=target_id).squeeze_(1)
    
    flag = (target==true_target)
    target += indexes[:, top]*flag - target*flag

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

def Soft_Mix(x_nat, x_ver, onehot, device):
    y_nat = label_smoothing(onehot, onehot.shape[1], 0.8)
    y_ver = label_smoothing(onehot, onehot.shape[1], 0.9)
    rand_lambda = torch.rand((x_nat.shape[0],1,1,1), device=device)
    x_ret = rand_lambda*x_nat + (1-rand_lambda)*x_ver
    rand_lambda = rand_lambda.squeeze(3).squeeze(2)
    y_ret = rand_lambda*y_nat + (1-rand_lambda)*y_ver
    return x_ret, y_ret

def All_Mix(inputs_1, inputs_2, y_1, y_2, device):
    rand_lambda = torch.rand((inputs_1.shape[0],1,1,1), device=device)
    x_ret = rand_lambda*inputs_1 + (1-rand_lambda)*inputs_2
    rand_lambda = rand_lambda.squeeze(3).squeeze(2)
    y_ret = rand_lambda*y_1 + (1-rand_lambda)*y_2
    return x_ret, y_ret

def Uncert_Mix(inputs_1, inputs_2, y_1, y_2, device):
    rand_lambda = torch.rand((inputs_1.shape[0],1,1,1), device=device)
    x_ret = rand_lambda*inputs_1 + (1-rand_lambda)*inputs_2
    
    rand_lambda = rand_lambda.squeeze(3).squeeze(2)
    y_mix = rand_lambda*y_1 + (1-rand_lambda)*y_2

    uncert = torch.abs(rand_lambda-0.5)*2
    y_uncert = label_smoothing(y_mix, y_mix.shape[1], 0.9)
    y_ret = uncert*y_uncert + (1-uncert)*y_uncert
    return x_ret, y_ret

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
    
    def final_shower(self, epoch_idx):
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
            # writer.add_scalar("clean_Acc", avg_acc, epoch_idx)
            # writer.add_scalar("clean_Loss", avg_loss, epoch_idx)

        # std adv test
        avg_loss, acc_sum, acc_count = self.std_adv_eval("{}/{}".format(epoch_idx, self.epochs))
        avg_loss = collect(avg_loss, self.device)
        avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            tqdm.write("Eval (Adver) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))
            # writer.add_scalar("adv_Acc", avg_acc, epoch_idx)
            # writer.add_scalar("adv_Loss", avg_loss, epoch_idx)
        
        # std adv test
        avg_loss, acc_sum, acc_count = self.std_adv100_eval("{}/{}".format(epoch_idx, self.epochs))
        avg_loss = collect(avg_loss, self.device)
        avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            tqdm.write("Eval (Adver) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))
        
        # Lipz test
        # std_lipz = self.std_lipz_eval()
        adv_lipz = self.adv_lipz_eval()
        if torch.distributed.get_rank() == 0:
            tqdm.write("Eval (Lipz) {}/{}, adv Lipz. {:.6f}".format(epoch_idx, self.epochs, adv_lipz))
            # writer.add_scalar("std_Lipz", std_lipz, epoch_idx)
            # writer.add_scalar("adv_Lipz", adv_lipz, epoch_idx)

    def wo_tar_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            _model_freeze(self.model)
            inputs = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            _model_unfreeze(self.model)
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
            
            _model_freeze(self.model)
            inputs = untarget_attack(self.attacker, inputs, labels)
            _model_unfreeze(self.model)
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

    def mul_vertex_tar_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            _model_freeze(self.model)
            adv_inputs_0 = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            adv_inputs_1 = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            adv_inputs_2 = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            adv_inputs_3 = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            inputs = torch.cat((
                Plain_Mix(adv_inputs_0, inputs, self.device), Plain_Mix(adv_inputs_1, inputs, self.device),
                Plain_Mix(adv_inputs_2, inputs, self.device), Plain_Mix(adv_inputs_3, inputs, self.device),
            ), dim=0)

            labels = F.one_hot(labels, self.num_class).float()
            labels = torch.cat((labels,labels,labels,labels), dim=0)
            _model_unfreeze(self.model)

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

    def mmc_vertex_tar_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            _model_freeze(self.model)
            adv_inputs = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            _model_unfreeze(self.model)
            inputs = Plain_Mix(adv_inputs, inputs, self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()

    def mmc_vertex_untar_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            _model_freeze(self.model)
            adv_inputs = untarget_attack(self.attacker, inputs, labels)
            _model_unfreeze(self.model)
            inputs = Plain_Mix(adv_inputs, inputs, self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
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
            
            _model_freeze(self.model)
            adv_inputs = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            _model_unfreeze(self.model)
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
    
    def reg_vertex_tar_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            _model_freeze(self.model)
            adv_inputs = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            _model_unfreeze(self.model)
            inputs = Plain_Mix(adv_inputs, inputs, self.device)
            # labels = F.one_hot(labels, self.num_class).float()
            
            features = self.model.module.encoder(inputs)
            outputs = self.model.module.classifier(features)
            reg = torch.mean(torch.norm(features, dim=1))/10
            loss = 0.1*(reg + 1/reg) + nn.functional.cross_entropy(outputs, labels)

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
            
            _model_freeze(self.model)
            adv_inputs = untarget_attack(self.attacker, inputs, labels)
            _model_unfreeze(self.model)
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
            
            _model_freeze(self.model)
            adv_inputs_1 = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            adv_inputs_2 = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            _model_unfreeze(self.model)
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
            
            _model_freeze(self.model)
            adv_inputs_1 = untarget_attack(self.attacker, inputs, labels)
            adv_inputs_2 = untarget_attack(self.attacker, inputs, labels)
            _model_unfreeze(self.model)
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
            
            _model_freeze(self.model)
            adv_inputs_1 = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            adv_inputs_2 = target_attack(self.attacker, inputs, labels, self.num_class, self.device)
            _model_unfreeze(self.model)
            
            # adv_inputs_1 = Plain_Mix(adv_inputs_1, adv_inputs_2, self.device)
            # inputs = Plain_Mix(adv_inputs_1, inputs, self.device)

            # adv_inputs_1 = Plain_Mix(adv_inputs_1, inputs, self.device)
            # adv_inputs_2 = Plain_Mix(adv_inputs_2, inputs, self.device)
            # inputs = Plain_Mix(adv_inputs_1, adv_inputs_2, self.device)

            # labels = F.one_hot(labels, self.num_class).float()
            onehot = F.one_hot(labels, self.num_class).float()
            adv_inputs_1, labels_1 = Soft_Mix(inputs, adv_inputs_1, onehot, self.device)
            adv_inputs_2, labels_2 = Soft_Mix(inputs, adv_inputs_2, onehot, self.device)
            inputs, labels = All_Mix(adv_inputs_1, adv_inputs_2, labels_1, labels_2, self.device)
            
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
    
    def top_10_step(self, progress, top=10):
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs, labels in self.train_loader:

            # batchSize = labels_0.shape[0]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.model.eval()
            _model_freeze(self.model)

            output = self.model(inputs)
            _, indexes = torch.sort(output, dim=1)
            adv_inputs = top_attack(self.attacker, inputs, labels, indexes[:,0:top+1], self.device, top=top)

            inputs = Plain_Mix(adv_inputs, inputs, self.device)
            labels = F.one_hot(labels, self.num_class).float()
            _model_unfreeze(self.model)
            
            self.model.train()
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
            
            _model_freeze(self.model)
            adv_inputs_1 = untarget_attack(self.attacker, inputs, labels)
            adv_inputs_2 = untarget_attack(self.attacker, inputs, labels)
            _model_unfreeze(self.model)

            # adv_inputs_1 = Plain_Mix(adv_inputs_1, adv_inputs_2, self.device)
            # inputs = Plain_Mix(adv_inputs_1, inputs, self.device)
            
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
        _model_freeze(self.model)
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
        
        _model_unfreeze(self.model)
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)
    
    def std_adv_eval(self, progress):
        self.model.eval()
        _model_freeze(self.model)
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

        _model_unfreeze(self.model)
        return (loss_meter.report(), accuracy_meter.sum, accuracy_meter.count)

    def std_adv100_eval(self, progress):
        self.model.eval()
        _model_freeze(self.model)
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter()

        pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = untarget_attack(self.lipz_attacker, data, target)

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

    def adv_lipz_eval(self):
        self.model.eval()
        _model_freeze(self.model)

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
        
        _model_unfreeze(self.model)
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
    
    def reg_vertex_tar(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.reg_vertex_tar_step("{}/{}".format(epoch_idx, self.epochs))

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
    
    def top_10(self, writer, top):
        ## Add a Writer
        # self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.top_10_step("{}/{}".format(epoch_idx, self.epochs), top)

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))

    def mul_vertex_tar(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.mul_vertex_tar_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))

    def mmc_vertex_tar(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.mmc_vertex_tar_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
        self.final_shower(epoch_idx+2)
    
    def mmc_vertex_untar(self, writer):
        ## Add a Writer
        self.add_shower(0)
        for epoch_idx in range(self.epochs):

            avg_loss = self.mmc_vertex_untar_step("{}/{}".format(epoch_idx, self.epochs))

            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_shower(epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
        self.final_shower(epoch_idx+2)