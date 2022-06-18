import glob
import os

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T

def turn_sub(cifar100: datasets.CIFAR100, subset):
    new_data = []
    new_label = []
    for data, label in zip(cifar100.data, cifar100.targets):
        if label in range(subset):
            new_data.append(data)
            new_label.append(label)
    cifar100.data = new_data
    cifar100.targets = new_label

class MNIST(Dataset):
    def __init__(self, dataroot, transform=T.Compose([]), train=True, max_n_per_class=6000):
        self.dataroot_param = dataroot
        self.dataroot = dataroot
        self.train = train

        self.max_n_per_class = max_n_per_class

        self.transform = transform

        self.data = datasets.MNIST(root=self.dataroot, train=train, transform=self.transform, download=True)

        self.classes = self.data.classes
        self.class_num = len(self.data.classes)
        self.class_to_idx = self.data.class_to_idx
        self.idx_to_class = {self.class_to_idx[cls]:cls for cls in self.class_to_idx}

        self.subset_mask = np.array(self.data.targets)
        for i in range(10):
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
    
    def __getitem__(self, idx):
        return self.data.__getitem__(idx)
        idx_in_ori_data = self.subset_indices[idx]
        return self.data.__getitem__(idx_in_ori_data)

    def __len__(self):
        return len(self.subset_indices)

    def __repr__(self):
        repr = """MNIST Dataset(subset):
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

class FashionMNIST(Dataset):
    def __init__(self, dataroot, transform=T.Compose([]), train=True, max_n_per_class=6000):
        self.dataroot_param = dataroot
        self.dataroot = dataroot
        self.train = train

        self.max_n_per_class = max_n_per_class

        self.transform = transform

        self.data = datasets.FashionMNIST(root=self.dataroot, train=train, transform=self.transform, download=True)

        self.classes = self.data.classes
        self.class_num = len(self.data.classes)
        self.class_to_idx = self.data.class_to_idx
        self.idx_to_class = {self.class_to_idx[cls]:cls for cls in self.class_to_idx}

        self.subset_mask = np.array(self.data.targets)
        for i in range(10):
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
    
    def __getitem__(self, idx):
        return self.data.__getitem__(idx)
        idx_in_ori_data = self.subset_indices[idx]
        return self.data.__getitem__(idx_in_ori_data)

    def __len__(self):
        return len(self.subset_indices)

    def __repr__(self):
        repr = """FashionMNIST Dataset(subset):
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

class Cifar10(Dataset):
    def __init__(self, dataroot, transform=T.Compose([]), train=True, max_n_per_class=6000):
        self.dataroot_param = dataroot
        self.dataroot = dataroot
        self.train = train

        self.max_n_per_class = max_n_per_class

        self.transform = transform

        self.data = datasets.CIFAR10(root=self.dataroot, train=train, transform=self.transform, download=True)

        self.classes = self.data.classes
        self.class_num = len(self.data.classes)
        self.class_to_idx = self.data.class_to_idx
        self.idx_to_class = {self.class_to_idx[cls]:cls for cls in self.class_to_idx}

        self.subset_mask = np.array(self.data.targets)
        for i in range(10):
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
    
    def __getitem__(self, idx):
        return self.data.__getitem__(idx)
        idx_in_ori_data = self.subset_indices[idx]
        return self.data.__getitem__(idx_in_ori_data)

    def __len__(self):
        return len(self.subset_indices)

    def __repr__(self):
        repr = """Cifar10 Dataset(subset):
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

class Cifar100(Dataset):
    def __init__(self, dataroot, transform=T.Compose([]), train=True, max_n_per_class=600):
        self.dataroot_param = dataroot
        self.dataroot = dataroot
        self.train = train

        self.max_n_per_class = max_n_per_class

        self.transform = transform

        self.data = datasets.CIFAR100(root=self.dataroot, train=train, transform=self.transform, download=True)

        self.classes = self.data.classes
        self.class_num = len(self.data.classes)
        self.class_to_idx = self.data.class_to_idx
        self.idx_to_class = {self.class_to_idx[cls]:cls for cls in self.class_to_idx}

        self.subset_mask = np.array(self.data.targets)
        for i in range(100):
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
    
    def __getitem__(self, idx):
        return self.data.__getitem__(idx)
        idx_in_ori_data = self.subset_indices[idx]
        return self.data.__getitem__(idx_in_ori_data)

    def __len__(self):
        return len(self.subset_indices)

    def __repr__(self):
        repr = """Cifar100 Dataset(subset):
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

class subCifar100(Dataset):
    def __init__(self, dataroot, transform=T.Compose([]), train=True, subset=10, max_n_per_class=600):
        self.dataroot_param = dataroot
        self.dataroot = dataroot
        self.train = train

        self.max_n_per_class = max_n_per_class

        self.transform = transform

        self.data = datasets.CIFAR100(root=self.dataroot, train=train, transform=self.transform, download=True)
        turn_sub(self.data, subset)

        self.classes = self.data.classes
        self.class_num = len(self.data.classes)
        self.class_to_idx = self.data.class_to_idx
        self.idx_to_class = {self.class_to_idx[cls]:cls for cls in self.class_to_idx}

        self.subset_mask = np.array(self.data.targets)
        for i in range(100):
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
    
    def __getitem__(self, idx):
        return self.data.__getitem__(idx)
        idx_in_ori_data = self.subset_indices[idx]
        return self.data.__getitem__(idx_in_ori_data)

    def __len__(self):
        return len(self.subset_indices)

    def __repr__(self):
        repr = """Cifar100 Dataset(subset):
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr


class SVHN(Dataset):
    def __init__(self, dataroot, transform=T.Compose([]), train=True, max_n_per_class=60000):
        self.dataroot_param = dataroot
        self.dataroot = dataroot
        self.train = train
        if train:
            self.split = 'train'
        else:
            self.split = 'test'

        self.max_n_per_class = max_n_per_class

        self.transform = transform

        self.data = datasets.SVHN(root=self.dataroot, split=self.split, transform=self.transform, download=True)

        # self.classes = self.data.classes
        self.class_num = 10
        # self.class_to_idx = self.data.class_to_idx
        # self.idx_to_class = {self.class_to_idx[cls]:cls for cls in self.class_to_idx}

        self.subset_mask = np.array(self.data.labels)
        for i in range(10):
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
    
    def __getitem__(self, idx):
        return self.data.__getitem__(idx)
        idx_in_ori_data = self.subset_indices[idx]
        return self.data.__getitem__(idx_in_ori_data)

    def __len__(self):
        return len(self.subset_indices)

    def __repr__(self):
        repr = """Cifar10 Dataset(subset):
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr


class ImageNet(Dataset):
    """ ImageNet dataset with subset and MAX #data-per-class settings.
    If use default parameters, it will just return a dataset with all ImageNet data.
    Otherwise, it will return a subset of ImageNet dataset.
    """
    def __init__(self, dataroot, transform=T.Compose([]), train=True, subset=1000, max_n_per_class=1000):
        # Initial parameters
        self.dataroot_param = dataroot
        self.dataroot = os.path.join(dataroot, "ilsvrc2012")
        self.train = train

        # Class number of subset. Take top-N classes as a subset(Torchvision official implementation sorting).
        self.subset = subset
        # Max number of data per class. If it was set more than the total number of that class, all the data will be taken.
        # Otherwise, it will take top-N data of that class(Torchvision official implementation sorting).
        self.max_n_per_class = max_n_per_class

        self.transform = transform
        
        self.data = datasets.ImageNet(root=self.dataroot, split='train' if train else 'val', transform=self.transform)
        
        # Metadata of dataset
        self.classes = self.data.classes
        self.class_num = len(self.data.classes)
        self.idx_to_class = {i:self.data.classes[i][0] for i in range(self.class_num)}
        self.class_to_idx = {}
        for i in self.idx_to_class:
            classname = self.idx_to_class[i]
            while classname in self.class_to_idx:
                classname += '_'
            self.class_to_idx[classname] = i
        
        # Subset process.
        if isinstance(subset, int):
            self.class_subset = list(range(subset))
        else:
            self.class_subset = list(subset)

        self.mapping = {i: self.class_subset[i] for i in range(len(self.class_subset))}
        self._rev_mapping = {self.mapping[i]: i for i in self.mapping}
        self._rev_mapping = np.array([self._rev_mapping[i] if i in self.class_subset else -1 for i in range(1000)])
        target_mapping = lambda x: self._rev_mapping[x]

        self.subset_mask = np.array(self.data.targets)
        for i in self.class_subset:
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
        self.class_selection = np.where(np.in1d(np.array(self.data.targets), np.array(self.class_subset)) == 1)[0]
        self.subset_indices = np.intersect1d(self.subset_indices, self.class_selection)

        # Data and targets
        self.targets = list(target_mapping(np.array(self.data.targets)[self.subset_indices]))
        self.img_paths = list(np.array(self.data.imgs)[self.subset_indices][:, 0])

        # Metadata override.
        self.classes = [self.classes[i] for i in self.class_subset]
        self.class_num = len(self.classes)
        self.idx_to_class = {i: self.idx_to_class[i] for i in self.class_subset}
        self.class_to_idx = {self.idx_to_class[i]: i for i in self.idx_to_class}


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        target = self.targets[idx]

        return (img_tensor, target)

    def __repr__(self):
        repr = """ImageNet Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

class subImageNet(Dataset):
    """ ImageNet dataset with subset and MAX #data-per-class settings.
    If use default parameters, it will just return a dataset with all ImageNet data.
    Otherwise, it will return a subset of ImageNet dataset.
    """
    def __init__(self, dataroot, transform=T.Compose([]), train=True, subset=100, max_n_per_class=10000):
        # Initial parameters
        self.dataroot_param = dataroot
        self.dataroot = os.path.join(dataroot, "ilsvrc2012")
        self.train = train

        # Class number of subset. Take top-N classes as a subset(Torchvision official implementation sorting).
        self.subset = subset
        # Max number of data per class. If it was set more than the total number of that class, all the data will be taken.
        # Otherwise, it will take top-N data of that class(Torchvision official implementation sorting).
        self.max_n_per_class = max_n_per_class

        self.transform = transform
        
        self.data = datasets.ImageNet(root=self.dataroot, split='train' if train else 'val', transform=self.transform)
        
        # Metadata of dataset
        self.classes = self.data.classes
        self.class_num = len(self.data.classes)
        self.idx_to_class = {i:self.data.classes[i][0] for i in range(self.class_num)}
        self.class_to_idx = {}
        for i in self.idx_to_class:
            classname = self.idx_to_class[i]
            while classname in self.class_to_idx:
                classname += '_'
            self.class_to_idx[classname] = i
        
        # Subset process.
        if isinstance(subset, int):
            self.class_subset = list(range(subset))
        else:
            self.class_subset = list(subset)

        self.mapping = {i: self.class_subset[i] for i in range(len(self.class_subset))}
        self._rev_mapping = {self.mapping[i]: i for i in self.mapping}
        self._rev_mapping = np.array([self._rev_mapping[i] if i in self.class_subset else -1 for i in range(1000)])
        target_mapping = lambda x: self._rev_mapping[x]

        self.subset_mask = np.array(self.data.targets)
        for i in self.class_subset:
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
        self.class_selection = np.where(np.in1d(np.array(self.data.targets), np.array(self.class_subset)) == 1)[0]
        self.subset_indices = np.intersect1d(self.subset_indices, self.class_selection)

        # Data and targets
        self.targets = list(target_mapping(np.array(self.data.targets)[self.subset_indices]))
        self.img_paths = list(np.array(self.data.imgs)[self.subset_indices][:, 0])

        # Metadata override.
        self.classes = [self.classes[i] for i in self.class_subset]
        self.class_num = len(self.classes)
        self.idx_to_class = {i: self.idx_to_class[i] for i in self.class_subset}
        self.class_to_idx = {self.idx_to_class[i]: i for i in self.idx_to_class}


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        target = self.targets[idx]

        return (img_tensor, target)

    def __repr__(self):
        repr = """ImageNet Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr



# Tiny-ImageNet
class TinyImageNet(Dataset):
    def __init__(self, dataroot, transform=T.Compose([]), train=True, max_n_per_class=600):
        self.dataroot_param = dataroot
        self.dataroot = dataroot
        self.train = train

        self.max_n_per_class = max_n_per_class

        self.transform = transform

        self.data = datasets.ImageFolder(root=os.path.join(self.dataroot, 'train' if train else 'val'), transform=self.transform)

        self.classes = self.data.classes
        self.class_num = len(self.classes)
        self.class_to_idx = self.data.class_to_idx
        self.idx_to_class = {self.class_to_idx[cls]:cls for cls in self.class_to_idx}

        self.subset_mask = np.array(self.data.targets)
        for i in range(200):
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
    
    def __getitem__(self, idx):
        return self.data.__getitem__(idx)
        idx_in_ori_data = self.subset_indices[idx]
        return self.data.__getitem__(idx_in_ori_data)

    def __len__(self):
        return len(self.subset_indices)

    def __repr__(self):
        repr = """Tiny Imagenet Dataset(subset):
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr



# sub-ImageNet32
class subImageNet32(Dataset):
    def __init__(self, dataroot, transform=T.Compose([]), train=True, max_n_per_class=1000):
        self.dataroot_param = dataroot
        self.dataroot = os.path.join(dataroot, "Imagenet-32-100")
        self.train = train

        self.max_n_per_class = max_n_per_class

        self.transform = transform

        self.data = datasets.ImageFolder(root=os.path.join(self.dataroot, 'train' if train else 'val'), transform=self.transform)

        self.classes = self.data.classes
        self.class_num = len(self.classes)
        self.class_to_idx = self.data.class_to_idx
        self.idx_to_class = {self.class_to_idx[cls]:cls for cls in self.class_to_idx}

        self.subset_mask = np.array(self.data.targets)
        for i in range(200):
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
    
    def __getitem__(self, idx):
        return self.data.__getitem__(idx)
        idx_in_ori_data = self.subset_indices[idx]
        return self.data.__getitem__(idx_in_ori_data)

    def __len__(self):
        return len(self.subset_indices)

    def __repr__(self):
        repr = """Imagenet32 Dataset(subset):
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

import cv2
class CUB(Dataset):

    def __init__(self, path, train=True, transform=None, target_transform=None):
        self.class_num=200
        self.root = path
        self.is_train = train
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = {}
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id
        
        self.data_id = []
        if self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if int(is_train):
                        self.data_id.append(image_id)
        if not self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if not int(is_train):
                        self.data_id.append(image_id)

    def __len__(self):
        return len(self.data_id)
    
    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_id = self.data_id[index]
        class_id = int(self._get_class_by_id(image_id)) - 1
        path = self._get_path_by_id(image_id)
        image = cv2.imread(os.path.join(self.root, 'images', path))
        
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            class_id = self.target_transform(class_id)
        return image, class_id

    def _get_path_by_id(self, image_id):

        return self.images_path[image_id]
    
    def _get_class_by_id(self, image_id):

        return self.class_ids[image_id]