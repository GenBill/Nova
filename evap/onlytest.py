import torch

def onlytest(dataloader, criterion, student, device):
    student.eval()
    with torch.no_grad():
        running_loss = 0
        running_acc = 0
        n_samples = 0
        for batch_num, (inputs, labels) in enumerate(dataloader):
            batchSize = labels.shape[0]
            n_samples += batchSize
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = student(inputs)
            loss = criterion(outputs, labels)

            pred_top_1 = torch.topk(outputs, k=1, dim=1)[1]
            this_acc = pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()
        
            running_loss += loss.item() * batchSize
            running_acc += this_acc

    epoch_loss = running_loss / n_samples
    epoch_acc = running_acc / n_samples

    if torch.distributed.get_rank() == 0:
        print('Test Result ...')
        print('Loss : {:.6f}, Acc : {:.6f}'.format(epoch_loss, epoch_acc))

    return student