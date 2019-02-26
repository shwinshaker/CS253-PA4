import utils as ut
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
from music_dataloader import createLoaders
import numpy as np


# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

# load data
train_loader, val_loader, test_loader = createLoaders(extras=extras)


criterion = nn.CrossEntropyLoss()
