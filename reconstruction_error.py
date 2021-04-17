import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from six.moves.urllib.request import urlretrieve

from urllib.error import URLError
from urllib.error import HTTPError


def to_var(tensor, cuda=True):
    """Wraps a Tensor in a Variable, optionally placing it on the GPU.

        Arguments:
            tensor: A Tensor object.
            cuda: A boolean flag indicating whether to use the GPU.

        Returns:
            A Variable object, on the GPU if cuda==True.
    """
    if cuda:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


def get_zero_vector(dim):

    return to_var(torch.zeros(dim))


def get_emnist_loader(emnist_type):
    transform = transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    test = datasets.EMNIST(".", split=emnist_type,
                           train=False, download=True, transform=transform)

    test_dloader = DataLoader(
        dataset=test, batch_size=1, shuffle=False, num_workers=0)
    return test_dloader


def load_model(path) -> nn.Module:
    """
    Add Your
    """
    NotImplementedError


model = load_model("")
z = get_zero_vector(100)
print(z)
z_optimizer = optim.RMSprop(model.parameters(), 5e-5)
get_emnist_loader("letter")
