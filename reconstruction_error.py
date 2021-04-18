import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from six.moves.urllib.request import urlretrieve
from model_wgan import DCGenerator
from urllib.error import URLError
from urllib.error import HTTPError
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def log_to_tensorboard(iteration, losses):
    writer = SummaryWriter("./runs/")
    for key in losses:
        arr = losses[key]
        writer.add_scalar(f'loss/{key}', arr[-1], iteration)
    writer.close()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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
    """
    """
    return torch.zeros(1, dim).requires_grad_()


def get_emnist_loader(opts):
    transform = transforms.Compose([
        transforms.Scale(opts.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    test = datasets.EMNIST(".", split=opts.X,
                           download=True, transform=transform)

    test_dloader = DataLoader(
        dataset=test, batch_size=opts.batch_size, shuffle=True, num_workers=0)
    return test_dloader


def load_generator(opts) -> nn.Module:
    """

    """
    G_path = os.path.join(opts.load, 'G.pkl')
    print(G_path)
    G = DCGenerator(noise_size=opts.noise_size,
                    conv_dim=opts.g_conv_dim, spectral_norm=False)

    G.load_state_dict(torch.load(
        G_path, map_location=lambda storage, loc: storage))
    return G


def train(opts):
    G = load_generator(opts)
    z = get_zero_vector(opts.noise_size)

    z_optimizer = optim.SGD([z], lr=1e-4)
    test_data = get_emnist_loader(opts)
    train_iter = iter(test_data)
    iter_per_epoch = len(train_iter)

    criteria = nn.MSELoss()
    iteration = 0
    epoch = 0
    losses = {"reconstruction_loss": []}
    for i in range(opts.iterations):
        sample, target = train_iter.next()
        if iteration % iter_per_epoch == 0:
            epoch += 1
            train_iter = iter(test_data)
            print("epoch:", epoch)

        z_optimizer.zero_grad()
        Loss = criteria(G(z.unsqueeze(2).unsqueeze(3)), sample)
        Loss.backward()
        z_optimizer.step()
        if iteration % 1000 == 0:
            losses["reconstruction_loss"].append(Loss)
            log_to_tensorboard(iteration, losses)
            print('iteration', iteration, "loss", Loss)

        iteration += 1


if __name__ == "__main__":
    args = AttrDict()
    args_dict = {
        'image_size': 32,
        'g_conv_dim': 32,
        'noise_size': 100,
        'num_workers': 0,
        'iterations': 120000,
        'X': 'letters',
        'batch_size': 16,
        'load': "./pretrained_models/WGAN",
        'log_step': 100,
        'sample_every': 200,
        'checkpoint_every': 1000,
    }
    args.update(args_dict)
    train(args)
