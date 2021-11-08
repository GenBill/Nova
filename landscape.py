import torch
import torch.nn as nn
import pickle
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision
from model import resnet18_small
# from model import PreActResNet18
# from attacker import normalize

from utils import get_device_id

def RaderMacher(v, size):
    vec = torch.rand(size)
    vec[vec > v] = 1
    vec[vec < v] = -1
    vec[vec == v] = 0
    return vec


def landscape_show(model, X, target, loss_fn=nn.CrossEntropyLoss(reduction='none'),
                   lower=-0.25, upper=0.25, nb=50, zlim_lower=-2.3, zlim_uper=2.3, cmap=plt.get_cmap('rainbow'),
                   mean_std=None):
    model = model.eval()
    model = pickle.loads(pickle.dumps(model))
    for param in model.parameters():
        param.requires_grad = False
    # if mean_std is not None:
    #     X = normalize(X, mean_std[0], mean_std[1])
    X = Variable(X, requires_grad=True)
    loss = loss_fn(model(X), target)

    stdloss = loss.item()
    loss.backward()
    x = torch.sign(X.grad)
    y = RaderMacher(0.5, X.size())
    X.requires_grad = False

    fig = plt.figure()
    ax = Axes3D(fig)

    tr = upper - lower
    L = np.arange(lower, upper, tr / nb)
    R = np.arange(lower, upper, tr / nb)
    signX, signY = np.meshgrid(L, R)
    losses = np.zeros_like(signX)

    with torch.no_grad():
        for i in range(np.shape(signX)[0]):
            total_inp = X.repeat(nb, 1, 1, 1)
            for j in range(np.shape(signY)[0]):
                l = signX[i, j]
                r = signY[i, j]
                total_inp[j] = total_inp[j] + l * x + r * y
            loss = loss_fn(model(total_inp), target.repeat(nb))
            for j in range(np.shape(R)[0]):
                losses[i, j] = loss[j].item() - stdloss

    ax.plot_surface(signX, signY, losses, cmap=cmap)

    ax.set_zlim3d(zlim_lower, zlim_uper)
    plt.show()


def load_model(path, backbone):
    ckp = torch.load(path, map_location='cpu')
    if 'state_dict' in ckp:
        ckp = ckp['state_dict']
    model = backbone
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckp.items()})
    return model


def load_cifar10(data_root='../'):
    # cifar10_mean = np.array((0.4914, 0.4822, 0.4465))
    # cifar10_std = np.array((0.2471, 0.2435, 0.2616))
    # normalisze = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    # normalisze.requires_grad_(False)


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalisze
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_root,
                                            train=True,
                                            download=True,
                                            transform=transform_train)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root=data_root,
                                           train=False,
                                           download=True,
                                           transform=transform_test)
    image_size = 32
    num_classes = 10

    # cifar10_mean = torch.from_numpy(cifar10_mean[:, None, None]).\
    #     expand((3, image_size, image_size)).type(torch.FloatTensor)
    # cifar10_std = torch.from_numpy(cifar10_std[:, None, None]).\
    #     expand((3, image_size, image_size)).type(torch.FloatTensor)
    # cifar10_max = (1 - cifar10_mean) / cifar10_std
    # cifar10_min = (0 - cifar10_mean) / cifar10_std
    return (image_size, num_classes, trainset, testset) # , cifar10_max, cifar10_min)


if __name__ == '__main__':
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    device_id = get_device_id()
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'

    model = resnet18_small(n_class=10).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)

    checkpoint_path = 'checkpoint/multar-plain-SVHN-LRDL.pth'

    if torch.distributed.get_rank() == 0:
            print('\nEval on {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.state_dict())
    
    
    # model = load_model('models/model_final.pth', PreActResNet18(10))
    X = load_cifar10('~Datasets/cifar10')[3]   # ('D:/cifar10')[3]

    X, Y = X[1]
    X = X.unsqueeze(0)

    # cifar10_mean = torch.from_numpy(np.array((0.4914, 0.4822, 0.4465))).type(torch.FloatTensor) \
    #                    .cuda()[None, :, None, None].expand(1, 3, 32, 32)
    # cifar10_std = torch.from_numpy(np.array((0.2471, 0.2435, 0.2616))).type(torch.FloatTensor) \
    #                   .cuda()[None, :, None, None].expand(1, 3, 32, 32)

    # mean_std = (cifar10_mean, cifar10_std)
    landscape_show(model, X, torch.tensor([Y]).type(torch.LongTensor), nb=50, mean_std=None)