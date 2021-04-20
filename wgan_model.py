
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
import tarfile

import imageio
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


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


def upconv(in_channels, out_channels, kernel_size, stride=2, padding=2, batch_norm=True, spectral_norm=False):
    """Creates a upsample-and-convolution layer, with optional batch normalization.
    """
    layers = []
    if stride > 1:
        layers.append(nn.Upsample(scale_factor=stride))
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=1, padding=padding, bias=False)
    if spectral_norm:
        layers.append(SpectralNorm(conv_layer))
    else:
        layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=2, batch_norm=True, init_zero_weights=False, spectral_norm=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(
            out_channels, in_channels, kernel_size, kernel_size) * 0.001

    if spectral_norm:
        layers.append(SpectralNorm(conv_layer))
    else:
        layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(
            in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim, spectral_norm=False):
        super(DCGenerator, self).__init__()

        self.conv_dim = conv_dim
        self.relu = nn.ReLU()
        # BS X noise_size x 1 x 1 -> BS x 128 x 4 x 4
        self.linear_bn = upconv(100, conv_dim*4, 3)
        self.upconv1 = upconv(conv_dim*4, conv_dim*2, 5)
        self.upconv2 = upconv(conv_dim*2, conv_dim, 5)
        self.upconv3 = upconv(conv_dim, 1, 5, batch_norm=False)

        self.tanh = nn.Tanh()

    def forward(self, z):
        """Generates an image given a sample of random noise.

            Input
            -----
                z: BS x noise_size x 1 x 1   -->  BSx100x1x1 (during training)

            Output
            ------
                out: BS x channels x image_width x image_height  -->  BSx3x32x32 (during training)
        """
        batch_size = z.size(0)
        out = self.relu(self.linear_bn(z))  # BS x 128 x 4 x 4      conv_dim=32
        out = out.view(-1, self.conv_dim*4, 4, 4)
        out = self.relu(self.upconv1(out))  # BS x 64 x 8 x 8
        out = self.relu(self.upconv2(out))  # BS x 32 x 16 x 16
        out = self.tanh(self.upconv3(out))  # BS x 3 x 32 x 32
        out_size = out.size()
        if out_size != torch.Size([batch_size, 1, 32, 32]):
            raise ValueError(
                "expect {} x 3 x 32 x 32, but get {}".format(batch_size, out_size))
        return out


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim=64, spectral_norm=False):
        super(DCDiscriminator, self).__init__()

        self.conv1 = conv(in_channels=1, out_channels=conv_dim,
                          kernel_size=5, stride=2, spectral_norm=spectral_norm)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim*2,
                          kernel_size=5, stride=2, spectral_norm=spectral_norm)
        self.conv3 = conv(in_channels=conv_dim*2, out_channels=conv_dim*4,
                          kernel_size=5, stride=2, spectral_norm=spectral_norm)
        self.conv4 = conv(in_channels=conv_dim*4, out_channels=1, kernel_size=5,
                          stride=2, padding=1, batch_norm=False, spectral_norm=spectral_norm)

    def forward(self, x):
        batch_size = x.size(0)

        out = F.relu(self.conv1(x))    # BS x 64 x 16 x 16
        out = F.relu(self.conv2(out))    # BS x 64 x 8 x 8
        out = F.relu(self.conv3(out))    # BS x 64 x 4 x 4

        out = self.conv4(out).squeeze()
        out_size = out.size()
        if out_size != torch.Size([batch_size, ]):
            raise ValueError(
                "expect {} x 1, but get {}".format(batch_size, out_size))
        return out
