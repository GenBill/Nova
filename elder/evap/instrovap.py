import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler

def conti_CE(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), dim=1))

def feature_dist(fea_0, fea_1):
    return F.kl_div(fea_0.softmax(dim=-1).log(), fea_1.softmax(dim=-1), reduction='batchmean')
    # return nn.KLDivLoss(reduction='batchmean')(nn.Sigmoid()(fea_0), nn.Sigmoid()(fea_1))
    # return nn.MSELoss()(fea_0, fea_1)

def loss_to_weight(weight, eps=1e-8):
    sum = eps
    for i in weight:
        sum += i
    for i in weight:
        i /= sum
    return weight

def count_weighted_targets(targets, weight):
    Ret = 0
    model_num = len(targets)
    for i in range(model_num):
        # print(targets[i].shape, weight[i].unsqueeze(1).shape)
        Ret += targets[i] * weight[i]
    return Ret

# attacker = LinfPGD(model, epsilon=8/255, step=2/255, iterations=7, random_start=True)
def student_train(alpha, num_epoch, dataloader, criterion, teacher_list, student, optimizer, scheduler, device):
    '''
    including args
    args : lr, momentum, num_epoch, alpha
    '''
    # model_num = len(teacher_list)

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
            num_class = outputs_stu.shape[1]

            with torch.no_grad():
                outputs_tea = []
                weight = []
                for i, this_model in enumerate(teacher_list):
                    this_outputs = this_model(inputs).detach()
                    outputs_tea.append(this_outputs)
                    weight.append(1 - torch.topk(torch.log_softmax(this_outputs, dim=1), k=1, dim=1)[0])
                    # weight.append(criterion(this_outputs, labels).unsqueeze(1))
                    
                    # if output is wrong, weight = 0
                    pred_top_1 = torch.topk(this_outputs, k=1, dim=1)[1]
                    weight[i] *= pred_top_1.eq(labels.view_as(pred_top_1)).int()

            weight = loss_to_weight(weight)
            loss_plain = torch.mean(criterion(outputs_stu, labels))
            plain_targets = F.one_hot(labels, num_classes=num_class)
            soft_targets = (1-alpha) * plain_targets + alpha * count_weighted_targets(outputs_tea, weight)
            # print(soft_targets)
            loss_main = torch.sqrt(nn.MSELoss()(outputs_stu, soft_targets))
            # print(loss_plain.item(), loss_main.item())
            
            pred_top_1 = torch.topk(outputs_stu, k=1, dim=1)[1]
            this_acc = pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()

            loss_main.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss_main.item() * batchSize
            running_loss_plain += loss_plain.item() * batchSize
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
