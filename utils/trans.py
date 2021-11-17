import torch
import random

class Onepixel():
    def __init__(self, size_x=32, size_y=32):
        self.size_x = size_x
        self.size_y = size_y

    def __call__(self, tensor):
        x = random.randint(0, self.size_x-1)
        y = random.randint(0, self.size_y-1)

        tensor[0,x,y] = random.randint(0, 1)
        tensor[1,x,y] = random.randint(0, 1)
        tensor[2,x,y] = random.randint(0, 1)

        return tensor

class addNoise():
    def __init__(self, eps=8/255):
        self.eps = 2*eps

    def __call__(self, tensor):
        rand = (torch.rand(tensor.shape)-0.5) *self.eps
        return tensor + rand