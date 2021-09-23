import torch
import math
from tqdm import tqdm

def std_lipz(net, dataloader, device, rand_times=64, eps=1e-2):
    net.eval()
    all_Lipz = 0
    sample_size = len(dataloader.dataset)

    for batch_num, (example_0, _) in enumerate(tqdm(dataloader)):
        if batch_num==0:
            rand_shape = list(example_0.shape)
            rand_shape[0] = rand_times
        
        # while True:
        rand_vector = eps * (2*torch.rand(rand_shape)-1).to(device)
        example_0 = example_0.to(device)
        example_1 = example_0 + rand_vector

        # do a forward pass on the example
        pred_0 = net(example_0)
        pred_1 = net(example_1)
        
        # diff : rand_times * class_num
        # rand_vector : rand_times * 3 * 28 * 28
        diff = pred_1 - pred_0
        leng_diff = torch.sum(torch.abs(diff), dim=1)
        rand_vector = torch.abs(rand_vector).view(rand_times, -1)
        leng_vector = torch.max(rand_vector, dim=1).values
        
        # Count diff / rand_vector
        Ret = leng_diff / leng_vector
        Local_Lipz = torch.max(Ret, dim=0).values.item()

        
        '''
        # Count diff / rand_vector
        diff_list = torch.abs(diff[:, 0:1] / rand_vector) **2
        for i in range(1, class_num):
            diff_list += torch.abs(diff[:, i:i+1] / rand_vector) **2
        length_diff = torch.sqrt(torch.sum(diff_list, dim=1))

        # avoid divide 0
        zero = torch.zeros_like(length_diff)
        length_diff = torch.where(length_diff == math.inf, zero, length_diff)
        length_diff = torch.where(length_diff == math.nan, zero, length_diff)
        
        Local_Lipz = torch.max(length_diff, dim=0).values.item()
        if Local_Lipz>0:
            break
        '''
        
        all_Lipz += Local_Lipz / sample_size

    return all_Lipz


def adv_lipz(net, dataloader, attacker, device):
    net.eval()
    all_Lipz = 0
    sample_size = len(dataloader.dataset)

    for example_0, labels in tqdm(dataloader):
        example_0 = example_0.to(device)
        example_1 = attacker.attack(example_0, labels)

        # do a forward pass on the example
        pred_0 = net(example_0)
        pred_1 = net(example_1)
        
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
