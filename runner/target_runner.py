from torch._C import Size
from tqdm.auto import tqdm
from utils import AverageMeter
from utils import collect

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import copy

# from advertorch.attacks import LinfPGDAttack

# from runner.my_Rand import rain_Rand as rain_Rand
# from runner.my_Rand import noRand_ones as tower_Rand
from runner.my_Rand import softower_Rand as tower_Rand

# from runner.my_Rand import irain_Rand as rain_Rand
# from runner.my_Rand import itower_Rand as tower_Rand

from runner.my_Rand import brain_Rand as rain_Rand
# from runner.my_Rand import btower_Rand as tower_Rand


def soft_loss(pred, soft_targets):
    # softmax_pred = nn.LogSoftmax(dim=1)(pred)
    return torch.sqrt(nn.MSELoss()(pred, soft_targets))

# def soft_loss(pred, soft_targets):
#     logsoftmax = nn.LogSoftmax(dim=1)
#     return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), dim=1))

def target_attack(adversary, inputs, true_target, num_class, device, gamma=0.5):
    target = torch.randint(low=0, high=num_class-1, size=true_target.shape, device=device)
    # Ensure target != true_target
    target += (target >= true_target).int()
    adversary.targeted = True

    ## Reach Tower
    # adv_inputs = 2*adversary.perturb(inputs, target).detach() - inputs

    ## Reach Plain
    adv_inputs = adversary.perturb(inputs, target).detach()
    
    ## Rand Gamma
    rand_lambda = tower_Rand((true_target.shape[0],1,1,1), device=device)
    ret_inputs = rand_lambda*adv_inputs + (1-rand_lambda)*inputs
    
    rand_lambda.squeeze_(3).squeeze_(2)
    rand_lambda *= gamma
    ret_target = rand_lambda*F.one_hot(target, num_class).to(device) + (1-rand_lambda)*F.one_hot(true_target, num_class).to(device)
    
    ## Still Gamma
    # ret_inputs = adv_inputs
    # ret_target = gamma*F.one_hot(target, num_class).to(device) + (1-gamma)*F.one_hot(true_target, num_class).to(device)
    
    return ret_inputs, ret_target

def untarget_attack(adversary, inputs, true_target):
    adversary.targeted = False
    return adversary.perturb(inputs, true_target).detach()


class TargetRunner():
    def __init__(self, epochs, model, train_loader, shadow_loader, test_loader, criterion, optimizer, scheduler, attacker, num_class, device, gamma=0.5):
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
        self.gamma = gamma

        self.desc = lambda status, progress: f"{status}: {progress}"
        
    def add_writer(self, writer, epoch_idx):
        # clean test
        avg_loss, acc_sum, acc_count = self.clean_eval("{}/{}".format(epoch_idx, self.epochs))
        avg_loss = collect(avg_loss, self.device)
        avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        if torch.distributed.get_rank() == 0:
            tqdm.write("Eval (Clean) {}/{}, Loss avg. {:.6f}, Acc. {:.6f}".format(epoch_idx, self.epochs, avg_loss, avg_acc))
            writer.add_scalar("clean_Acc", avg_acc, epoch_idx)
            writer.add_scalar("clean_Loss", avg_loss, epoch_idx)

        # adv test
        avg_loss, acc_sum, acc_count = self.adv_eval("{}/{}".format(epoch_idx, self.epochs))
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
            rand_vector = rain_Rand((batchSize, 1), device=self.device)
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

    def adv_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for item in self.train_loader:
            inputs_0, labels_0 = item

            # batchSize = labels_0.shape[0]
            inputs_0 = inputs_0.to(self.device)
            labels_0 = labels_0.to(self.device)
            inputs, labels = target_attack(self.attacker, inputs_0, labels_0, self.num_class, self.device, self.gamma)
            
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

    
    def shades_adv_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for item1, item2 in zip(self.train_loader, self.shadow_loader):

            inputs_0, labels_0 = item1
            inputs_1, labels_1 = item2

            batchSize = labels_0.shape[0]
            inputs_0 = inputs_0.to(self.device)
            labels_0 = labels_0.to(self.device)
            inputs_0, labels_0 = target_attack(self.attacker, inputs_0, labels_0, self.num_class, self.device, self.gamma)
            
            inputs_1 = inputs_1.to(self.device)
            labels_1 = labels_1.to(self.device)
            inputs_1, labels_1 = target_attack(self.attacker, inputs_1, labels_1, self.num_class, self.device, self.gamma)

            # Create inputs & labels
            rand_vector = rain_Rand((batchSize, 1), device=self.device)
            inputs = inputs_0 * rand_vector.unsqueeze(2).unsqueeze(3) + inputs_1 * (1-rand_vector.unsqueeze(2).unsqueeze(3))
            labels = labels_0 * rand_vector + labels_1 * (1-rand_vector)
            del inputs_0, inputs_1, labels_0, labels_1, rand_vector
            
            # print(inputs.requires_grad, inputs.shape)
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
    

    def multar_adv_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for item1 in self.train_loader:

            inputs_0, labels_0 = item1
            batchSize = labels_0.shape[0]
            inputs_0 = inputs_0.to(self.device)
            labels_0 = labels_0.to(self.device)

            inputs_1, labels_1 = target_attack(self.attacker, inputs_0, labels_0, self.num_class, self.device, self.gamma)
            inputs_2, labels_2 = target_attack(self.attacker, inputs_0, labels_0, self.num_class, self.device, self.gamma)

            # Create inputs & labels
            rand_vector = rain_Rand((batchSize, 1), device=self.device)
            inputs = inputs_1 * rand_vector.unsqueeze(2).unsqueeze(3) + inputs_2 * (1-rand_vector.unsqueeze(2).unsqueeze(3))
            labels = labels_1 * rand_vector + labels_2 * (1-rand_vector)
            del inputs_0, inputs_1, inputs_2, labels_0, labels_1, labels_2, rand_vector
            
            # print(inputs.requires_grad, inputs.shape)
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
    
    def std_lipz_eval(self, rand_times=256, eps=1.0/255):
        self.model.eval()
        with torch.no_grad():
            all_Lipz = 0
            sample_size = len(self.test_loader.dataset)

            for batch_num, (example_0, _) in enumerate(self.test_loader):
                rand_shape = list(example_0.shape)
                example_0 = example_0.to(self.device)
                pred_0 = self.model(example_0)

                Local_Lipz = torch.zeros(rand_shape[0]).to(self.device)

                # while True:
                for i in range(rand_times):
                    rand_vector = eps * (2*torch.rand(rand_shape)-1).to(self.device)
                    pred_1 = self.model(example_0 + rand_vector)
                
                    # diff : rand_times * class_num
                    # rand_vector : rand_times * 3 * 28 * 28
                    diff = pred_1 - pred_0
                    leng_diff = torch.sum(torch.abs(diff), dim=1)
                    rand_vector = torch.abs(rand_vector).view(rand_shape[0], -1)
                    leng_vector = torch.max(rand_vector, dim=1).values
                
                    # Count diff / rand_vector
                    Ret = leng_diff / leng_vector
                    Local_Lipz = Ret * (Ret>Local_Lipz) + Local_Lipz * (Ret<=Local_Lipz)
                
                all_Lipz += torch.sum(Local_Lipz) / sample_size

        return all_Lipz

    def adv_lipz_eval(self):
        self.model.eval()
        all_Lipz = 0
        sample_size = len(self.test_loader.dataset)

        for example_0, labels in self.test_loader:
            labels = labels.to(self.device)
            example_0 = example_0.to(self.device)
            example_1 = untarget_attack(self.attacker, example_0, labels)

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
    
    def shades_train(self, adv=True):
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
                avg_loss = self.shades_adv_step("{}/{}".format(epoch_idx, self.epochs))
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
    
    def multar_train(self, writer, adv=True):
        # (avg_loss, acc_sum, acc_count) = self.adv_eval("Adv init")
        # avg_loss = collect(avg_loss, self.device)
        # avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        # if torch.distributed.get_rank() == 0:
        #     tqdm.write("Eval (Adver) init, Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))

        # (avg_loss, acc_sum, acc_count) = self.clean_eval("Clean init")
        # avg_loss = collect(avg_loss, self.device)
        # avg_acc = collect(acc_sum, self.device, mode='sum') / collect(acc_count, self.device, mode='sum')
        # if torch.distributed.get_rank() == 0:
        #     tqdm.write("Eval (Clean) init, Loss avg. {:.6f}, Acc. {:.6f}".format(avg_loss, avg_acc))
        
        ## Add a Writer
        # self.add_writer(writer, 0)
        for epoch_idx in range(self.epochs):
            if adv:
                avg_loss = self.multar_adv_step("{}/{}".format(epoch_idx, self.epochs))
            else:
                avg_loss = self.clean_step("{}/{}".format(epoch_idx, self.epochs))
            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                if adv:
                    tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
                else:
                    tqdm.write("Clean training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            # self.add_writer(writer, epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

            '''
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
            
            '''

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
