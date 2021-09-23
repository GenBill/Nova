import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler


def feature_dist(fea_0, fea_1):
    return F.kl_div(fea_0.softmax(dim=-1).log(), fea_1.softmax(dim=-1), reduction='batchmean')
    # return nn.KLDivLoss(reduction='batchmean')(nn.Sigmoid()(fea_0), nn.Sigmoid()(fea_1))
    # return nn.MSELoss()(fea_0, fea_1)

def count_loss_weight(loss, weight, eps=1e-8):
    model_num = len(loss)
    Ret = []
    for i in range(model_num):
        Ret.append(loss[i].detach() * weight[i])

    sumsum = Ret[0].detach()
    for i in range(1, model_num):
        sumsum += Ret[i].detach()

    for i in range(model_num):
        Ret[i] /= sumsum + eps
    
    return Ret

def student_train(alpha, num_epoch, dataloader, criterion, teacher_list, student, optimizer, scheduler, device):
    '''
    including args
    args : lr, momentum, num_epoch, alpha
    '''
    model_num = len(teacher_list)

    for this_model in teacher_list:
        this_model.eval()
    student.train()

    if torch.distributed.get_rank() == 0:
        print('Distilling by Rotavap ...')
    for epoch in range(num_epoch):
        running_loss = 0
        running_loss_plain = 0
        running_acc = 0
        n_samples = 0

        for batch_num, (inputs, labels) in enumerate(dataloader):
            batchSize = labels.shape[0]
            n_samples += batchSize
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs_stu = student(inputs)
            loss_0 = torch.mean(criterion(outputs_stu, labels))
            loss_plain = loss_0.item()
            
            with torch.no_grad():
                outputs_tea = []
                weight = []
                for i, this_model in enumerate(teacher_list):
                    this_outputs = this_model(inputs)
                    outputs_tea.append(this_outputs)
                    weight.append(criterion(this_outputs, labels))
                    
                    pred_top_1 = torch.topk(this_outputs, k=1, dim=1)[1]
                    weight[i] *= pred_top_1.eq(labels.view_as(pred_top_1)).squeeze().int()
            
            pred_top_1 = torch.topk(outputs_stu, k=1, dim=1)[1]
            this_acc = pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()

            loss_1 = []
            for this_fea in outputs_tea:
                loss_1.append(feature_dist(outputs_stu, this_fea))
            
            loss_main = loss_0 * (1-alpha)
            loss_weight = count_loss_weight(loss_1, weight)
            for i in range(model_num):
                loss_main += torch.mean(loss_weight[i]) * alpha

            loss_main.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss_main.item() * batchSize
            running_loss_plain += loss_plain * batchSize
            running_acc += this_acc

        epoch_loss = running_loss / n_samples
        epoch_loss_plain = running_loss_plain / n_samples
        epoch_acc = running_acc / n_samples

        if torch.distributed.get_rank() == 0:
            print('Epoch {}\nLoss : {:.6f}, Plain Loss : {:.6f}'.format(epoch, epoch_loss, epoch_loss_plain))
            print('Acc : {:.6f}'.format(epoch_acc))

    # return student

def evap(alpha, num_epoch, student, dataloader, criterion, model_list, optimizer, scheduler, device):
    '''
    including args
    args : lr, momentum, num_epoch, alpha
    '''
    # model_num = len(model_list)

    student_train(alpha, num_epoch, dataloader, criterion, model_list, student, optimizer, scheduler, device)
    # return student
