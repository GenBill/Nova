from tqdm.auto import tqdm
from utils import AverageMeter
from utils import collect

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from runner.my_Rand import rain_Rand, tower_Rand

# def soft_loss(pred, soft_targets):
#     return torch.sqrt(nn.MSELoss()(pred, soft_targets))

## L2 Loss
# def soft_loss(pred, soft_targets):
#     # return torch.mean(torch.sqrt(torch.mean((pred-soft_targets)**2, dim=1)), dim=0)
#     return torch.mean(torch.norm(pred-soft_targets, dim=1), dim=0)

def soft_loss(pred, soft_targets):
    # logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_targets * F.log_softmax(pred, dim=1), dim=1))

def label_smoothing(onehot, n_classes, factor):
    return onehot * factor + (onehot - 1) * ((factor - 1)/(n_classes - 1))

import numpy as np
def Get_Beta(size, device, alpha=1.0):
    rand_lambda = torch.zeros(size).to(device)
    for i in range(rand_lambda.shape[0]):
        rand_lambda[i,0,0,0] = np.random.beta(alpha, alpha)
    return rand_lambda

def Uncert_Mix(inputs_1, inputs_2, y_1, y_2, device, omega=0.9):
    rand_lambda = torch.rand((inputs_1.shape[0],1,1,1), device=device)
    # rand_lambda = Get_Beta((inputs_1.shape[0],1,1,1), device=device)
    x_ret = rand_lambda*inputs_1 + (1-rand_lambda)*inputs_2
    
    rand_lambda = rand_lambda.squeeze(3).squeeze(2)
    y_mix = rand_lambda*y_1 + (1-rand_lambda)*y_2

    uncert = 1-torch.abs(rand_lambda-0.5)*2

    # y_uncert = label_smoothing(y_mix, y_mix.shape[1], 0.9)
    y_uncert = label_smoothing(y_mix, y_mix.shape[1], omega)

    y_ret = uncert*y_uncert + (1-uncert)*y_mix
    return x_ret, y_ret

class ShadesRunner():
    def __init__(self, epochs, model, train_loader, shadow_loader, test_loader, criterion, optimizer, scheduler, attacker, num_class, device):
        self.device = device
        self.epochs = epochs
        self.eval_interval = 20

        self.model = model
        self.train_loader = train_loader
        self.shadow_loader = shadow_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.attacker = attacker
        self.num_class = num_class

        self.omega = 0.9

        self.desc = lambda status, progress: f"{status}: {progress}"
        
    def clean_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Clean train", progress))
        for item1, item2 in zip(self.train_loader, self.shadow_loader):
            inputs_0, labels_0 = item1
            inputs_1, labels_1 = item2
        
            batchSize = labels_0.shape[0]
            inputs_0 = inputs_0.to(self.device)
            labels_0 = labels_0.to(self.device)
            labels_0 = F.one_hot(labels_0, num_classes=self.num_class)
            
            inputs_1 = inputs_1.to(self.device)
            labels_1 = labels_1.to(self.device)
            labels_1 = F.one_hot(labels_1, num_classes=self.num_class)

            # Create inputs & labels
            inputs, labels = Uncert_Mix(inputs_0, inputs_1, labels_0, labels_1, self.device, self.omega)
            del inputs_0, inputs_1, labels_0, labels_1
            
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

    def adv_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for item1, item2 in zip(self.train_loader, self.shadow_loader):
            inputs_0, labels_0 = item1
            inputs_1, labels_1 = item2

            batchSize = labels_0.shape[0]
            inputs_0 = inputs_0.to(self.device)
            labels_0 = labels_0.to(self.device)

            inputs_0 = self.attacker.attack(inputs_0, labels_0)
            inputs_1 = self.attacker.attack(inputs_1, labels_1)
            labels_0 = F.one_hot(labels_0, num_classes=self.num_class)
            
            inputs_1 = inputs_1.to(self.device)
            labels_1 = labels_1.to(self.device)
            labels_1 = F.one_hot(labels_1, num_classes=self.num_class)

            # Create inputs & labels
            rand_vector = torch.rand((batchSize, 1), device=self.device)
            inputs = inputs_0 * rand_vector.unsqueeze(2).unsqueeze(3) + inputs_1 * (1-rand_vector.unsqueeze(2).unsqueeze(3))
            labels = labels_0 * rand_vector + labels_1 * (1-rand_vector)
            del inputs_0, inputs_1, labels_0, labels_1, rand_vector
            
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

    def double_adv_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for item1, item2 in zip(self.train_loader, self.shadow_loader):
            with torch.no_grad():
                inputs_0, targets_0 = item1
                inputs_1, targets_1 = item2

                batchSize = targets_0.shape[0]
                inputs_0 = inputs_0.to(self.device)
                targets_0 = targets_0.to(self.device)
                labels_0 = F.one_hot(targets_0, num_classes=self.num_class)
                
                inputs_1 = inputs_1.to(self.device)
                targets_1 = targets_1.to(self.device)
                labels_1 = F.one_hot(targets_1, num_classes=self.num_class)

                # Create inputs & labels
                rand_vector = torch.rand((batchSize, 1), device=self.device)
                inputs = inputs_0 * rand_vector.unsqueeze(2).unsqueeze(3) + inputs_1 * (1-rand_vector.unsqueeze(2).unsqueeze(3))
                labels = labels_0 * rand_vector + labels_1 * (1-rand_vector)
                del inputs_0, inputs_1, labels_0, labels_1, rand_vector
            
            # print(inputs.requires_grad, inputs.shape)
            inputs = self.attacker.double_attack(inputs, targets_0, targets_1)
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

    def mesa_adv_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for item1, item2 in zip(self.train_loader, self.shadow_loader):
            with torch.no_grad():
                inputs_0, targets_0 = item1
                inputs_1, targets_1 = item2

                batchSize = targets_0.shape[0]
                inputs_0 = inputs_0.to(self.device)
                targets_0 = targets_0.to(self.device)
                labels_0 = F.one_hot(targets_0, num_classes=self.num_class)
                
                inputs_1 = inputs_1.to(self.device)
                targets_1 = targets_1.to(self.device)
                labels_1 = F.one_hot(targets_1, num_classes=self.num_class)

                # Create inputs & labels
                rand_vector = torch.rand((batchSize, 1), device=self.device)
                inputs = inputs_0 * rand_vector.unsqueeze(2).unsqueeze(3) + inputs_1 * (1-rand_vector.unsqueeze(2).unsqueeze(3))
                labels = labels_0 * rand_vector + labels_1 * (1-rand_vector)
                del inputs_0, inputs_1, labels_0, labels_1, rand_vector
            
            # print(inputs.requires_grad, inputs.shape)
            inputs = self.attacker.mesa_attack(inputs, labels)
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
        with torch.no_grad():
            pbar = tqdm(total=len(self.test_loader), leave=False, desc=self.desc("Adv eval", progress))
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                data = self.attacker.attack(data, target)

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
                avg_loss, acc_sum, acc_count = self.adv_eval("{}/{}".format(epoch_idx, self.epochs))
                avg_loss = collect(avg_loss, self.device)
                avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
                if torch.distributed.get_rank() == 0:
                    tqdm.write("Eval (Adver) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))

                avg_loss, acc_sum, acc_count = self.clean_eval("{}/{}".format(epoch_idx, self.epochs))
                avg_loss = collect(avg_loss, self.device)
                avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
                if torch.distributed.get_rank() == 0:
                    tqdm.write("Eval (Clean) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))

    def Uncert_train(self, adv=True):
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
                avg_loss, acc_sum, acc_count = self.adv_eval("{}/{}".format(epoch_idx, self.epochs))
                avg_loss = collect(avg_loss, self.device)
                avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
                if torch.distributed.get_rank() == 0:
                    tqdm.write("Eval (Adver) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))

                avg_loss, acc_sum, acc_count = self.clean_eval("{}/{}".format(epoch_idx, self.epochs))
                avg_loss = collect(avg_loss, self.device)
                avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
                if torch.distributed.get_rank() == 0:
                    tqdm.write("Eval (Clean) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))

    def double_train(self, adv=True):
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
                avg_loss = self.double_adv_step("{}/{}".format(epoch_idx, self.epochs))
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
                avg_loss, acc_sum, acc_count = self.adv_eval("{}/{}".format(epoch_idx, self.epochs))
                avg_loss = collect(avg_loss, self.device)
                avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
                if torch.distributed.get_rank() == 0:
                    tqdm.write("Eval (Adver) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))

                avg_loss, acc_sum, acc_count = self.clean_eval("{}/{}".format(epoch_idx, self.epochs))
                avg_loss = collect(avg_loss, self.device)
                avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
                if torch.distributed.get_rank() == 0:
                    tqdm.write("Eval (Clean) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
    
    def mesa_train(self, adv=True):
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
                avg_loss = self.mesa_adv_step("{}/{}".format(epoch_idx, self.epochs))
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
                avg_loss, acc_sum, acc_count = self.adv_eval("{}/{}".format(epoch_idx, self.epochs))
                avg_loss = collect(avg_loss, self.device)
                avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
                if torch.distributed.get_rank() == 0:
                    tqdm.write("Eval (Adver) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))

                avg_loss, acc_sum, acc_count = self.clean_eval("{}/{}".format(epoch_idx, self.epochs))
                avg_loss = collect(avg_loss, self.device)
                avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
                if torch.distributed.get_rank() == 0:
                    tqdm.write("Eval (Clean) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))