from operator import mod
from re import M, X
from statistics import mode

import torch
from model.upernet_convnext_tiny import upernet_convnext_tiny
if __name__ == '__main__':
    data = torch.randn((1,1,1,1,1))
    print(data)
