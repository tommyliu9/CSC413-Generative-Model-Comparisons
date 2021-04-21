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
from urllib.error import URLError
from urllib.error import HTTPError
from wgan_model import load_generator, sample_noise, AttrDict
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.utils import make_grid
if __name__ == "__main__":
    args = AttrDict()
    args_dict = {
        'image_size': 28,
        'g_conv_dim': 32,
        'noise_size': 100,
        'num_workers': 0,
        'iterations': 200000,
        'X': 'letters',
        'batch_size': 20,
        'load': "./pretrained_models/WGAN",
        'log_step': 100,
        'sample_every': 200,
        'checkpoint_every': 1000,
    }
    args.update(args_dict)
    G  = load_generator(args)
    z = sample_noise(args.batch_size, 100)
    mean, std = np.array([0.5]), np.array([0.5])

    samples = G(z)
    t = transforms.Resize(28)
    
    samples = t(samples)
    print(samples.shape)
    viz = make_grid(samples, nrow=5, padding=2).numpy()*std + mean
    fig = plt.figure()
    fig, ax = plt.subplots(figsize= (8,8), dpi=100)
    ax.imshow(np.transpose(viz, (1,2,0)))
    ax.grid(False)
    fig.savefig('./wgan_examples.png')

