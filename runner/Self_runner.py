from tqdm.auto import tqdm
from utils import AverageMeter
import torch
from utils import collect

def _model_freeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=False

def _model_unfreeze(model) -> None:
    for param in model.parameters():
        param.requires_grad=True


def kd_loss_function(output, target_output, temperature):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()


def target_attack(adversary, inputs, true_target, num_class, device):
    target = torch.randint(low=0, high=num_class-1, size=true_target.shape, device=device)
    # Ensure target != true_target
    target += (target >= true_target).int()
    
    adversary.targeted = True
    adv_inputs = adversary.perturb(inputs, target).detach()
    return adv_inputs

def untarget_attack(adversary, inputs, true_target):
    adversary.targeted = False
    return adversary.perturb(inputs, true_target).detach()


class SelfRunner():
    def __init__(self, epochs, model, train_loader, test_loader, criterion, optimizer, scheduler, attacker, device, num_class=10, lamb=1.0):
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

        # self.ImageLoss = torch.nn.MSELoss()

        self.temperature = 3
        self.alpha = 0.1
        self.beta = 1e-6


        self.desc = lambda status, progress: f"{status}: {progress}"

    def clean_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Clean train", progress))
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # batch_size = target.shape[0]
            data, target = data.to(self.device), target.to(self.device)
            
            output, middle_output1, middle_output2, middle_output3, \
            final_fea, middle1_fea, middle2_fea, middle3_fea = self.model(data, train=True)

            ## Count CE Loss
            loss = self.criterion(output, target)
            middle1_loss = self.criterion(middle_output1, target)
            middle2_loss = self.criterion(middle_output2, target)
            middle3_loss = self.criterion(middle_output3, target)

            ## Count KL Loss
            temp4 = torch.softmax(output /self.temperature, dim=1)
            loss1by4 = kd_loss_function(middle_output1, temp4.detach(), self.temperature) * (self.temperature**2)
            loss2by4 = kd_loss_function(middle_output2, temp4.detach(), self.temperature) * (self.temperature**2)
            loss3by4 = kd_loss_function(middle_output3, temp4.detach(), self.temperature) * (self.temperature**2)
            
            ## Count Hints
            feature_loss_1 = feature_loss_function(middle1_fea, final_fea.detach()) 
            feature_loss_2 = feature_loss_function(middle2_fea, final_fea.detach()) 
            feature_loss_3 = feature_loss_function(middle3_fea, final_fea.detach()) 

            total_loss = (1 - self.alpha) * (loss + middle1_loss + middle2_loss + middle3_loss) + \
                        self.alpha * (loss1by4 + loss2by4 + loss3by4) + \
                        self.beta * (feature_loss_1 + feature_loss_2 + feature_loss_3)

            pbar.set_postfix_str("Loss {:.6f}".format(total_loss.item()))
            loss_meter.update(total_loss.item())
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()

    def adv_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # batch_size = target.shape[0]
            data, target = data.to(self.device), target.to(self.device)
            _model_freeze(self.model)
            data = untarget_attack(self.attacker, data, target)
            _model_unfreeze(self.model)
            
            output, middle_output1, middle_output2, middle_output3, \
            final_fea, middle1_fea, middle2_fea, middle3_fea = self.model(data, train=True)

            ## Count CE Loss
            loss = self.criterion(output, target)
            middle1_loss = self.criterion(middle_output1, target)
            middle2_loss = self.criterion(middle_output2, target)
            middle3_loss = self.criterion(middle_output3, target)

            ## Count KL Loss
            temp4 = torch.softmax(output /self.temperature, dim=1)
            loss1by4 = kd_loss_function(middle_output1, temp4.detach(), self.temperature) * (self.temperature**2)
            loss2by4 = kd_loss_function(middle_output2, temp4.detach(), self.temperature) * (self.temperature**2)
            loss3by4 = kd_loss_function(middle_output3, temp4.detach(), self.temperature) * (self.temperature**2)
            
            ## Count Hints
            feature_loss_1 = feature_loss_function(middle1_fea, final_fea.detach()) 
            feature_loss_2 = feature_loss_function(middle2_fea, final_fea.detach()) 
            feature_loss_3 = feature_loss_function(middle3_fea, final_fea.detach()) 

            total_loss = (1 - self.alpha) * (loss + middle1_loss + middle2_loss + middle3_loss) + \
                        self.alpha * (loss1by4 + loss2by4 + loss3by4) + \
                        self.beta * (feature_loss_1 + feature_loss_2 + feature_loss_3)

            pbar.set_postfix_str("Loss {:.6f}".format(total_loss.item()))
            loss_meter.update(total_loss.item())
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()
    
    def emadv_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # batch_size = target.shape[0]
            data, target = data.to(self.device), target.to(self.device)
            _model_freeze(self.model)
            data = untarget_attack(self.attacker, data, target)
            _model_unfreeze(self.model)
            
            output, middle_output1, middle_output2, middle_output3, \
            final_fea, middle1_fea, middle2_fea, middle3_fea = self.model(data, train=True)

            ## Count CE Loss
            loss = self.criterion(output, target)
            middle1_loss = self.criterion(middle_output1, target)
            middle2_loss = self.criterion(middle_output2, target)
            middle3_loss = self.criterion(middle_output3, target)

            ## Count KL Loss
            temp4 = (output+middle_output1+middle_output2+middle_output3)/4
            temp4 = torch.softmax(temp4/self.temperature, dim=1)
            loss0by4 = kd_loss_function(output, temp4.detach(), self.temperature) * (self.temperature**2)
            loss1by4 = kd_loss_function(middle_output1, temp4.detach(), self.temperature) * (self.temperature**2)
            loss2by4 = kd_loss_function(middle_output2, temp4.detach(), self.temperature) * (self.temperature**2)
            loss3by4 = kd_loss_function(middle_output3, temp4.detach(), self.temperature) * (self.temperature**2)
            
            ## Count Hints
            final_hints = (final_fea+middle1_fea+middle2_fea+middle3_fea)/4
            feature_loss_0 = feature_loss_function(final_fea, final_hints.detach()) 
            feature_loss_1 = feature_loss_function(middle1_fea, final_hints.detach()) 
            feature_loss_2 = feature_loss_function(middle2_fea, final_hints.detach()) 
            feature_loss_3 = feature_loss_function(middle3_fea, final_hints.detach()) 

            total_loss = (1 - self.alpha) * (loss + middle1_loss + middle2_loss + middle3_loss) + \
                        self.alpha * (loss0by4 + loss1by4 + loss2by4 + loss3by4) + \
                        self.beta * (feature_loss_0 + feature_loss_1 + feature_loss_2 + feature_loss_3)

            pbar.set_postfix_str("Loss {:.6f}".format(total_loss.item()))
            loss_meter.update(total_loss.item())
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()

    def adv_target_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            _model_freeze(self.model)
            data = target_attack(self.attacker, data, target, self.num_class, self.device)
            _model_unfreeze(self.model)

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
    
    def emtrain(self, adv=True):
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
                avg_loss = self.emadv_step("{}/{}".format(epoch_idx, self.epochs))
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
    
    def train_target(self, adv=True):
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
                avg_loss = self.adv_target_step("{}/{}".format(epoch_idx, self.epochs))
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
