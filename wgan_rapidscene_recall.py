
import PySimpleGUI as sg
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from model_wgan import DCGenerator
from wgan_reconstruction_error import AttrDict, load_generator
import imageio
from torchvision import transforms
from io import BytesIO
import random
from torchvision import datasets
import base64


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
    return (torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


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


def convert_to_tensor_to_im(im):
    imagify = transforms.ToPILImage()

    buffered = BytesIO()
    im = im.squeeze()
    im = 180*im
    im = imagify(im).convert('L')
    im.save(buffered, format='png')
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


def get_random_image():
    pass


if __name__ == "__main__":
    args = AttrDict()
    args_dict = {
        'image_size': 32,
        'g_conv_dim': 32,
        'noise_size': 100,
        'num_workers': 0,
        'iterations': 140000,
        'X': 'letters',
        'batch_size': 25,
        'load': "./pretrained_models/WGAN",
        'log_step': 100,
        'sample_every': 200,
        'checkpoint_every': 1000,
    }
    args.update(args_dict)

    transform = transforms.Compose([
        transforms.Scale(args.image_size),
        transforms.ToTensor()
    ])
    test = datasets.EMNIST(".", split=args.X,
                           download=True, transform=transform)

    G = load_generator(args)
    fake_samples = G(sample_noise(25, args.noise_size)).detach()
    lengths = [25, len(test) - 25]
    real_samples_idx, _ = random_split(test, lengths)

    fake_samples.requires_grad = False
    mean, std = np.array([0.5]), np.array([0.5])

    real_loader = DataLoader(real_samples_idx)
    real_iter = iter(real_loader)

    # Create an event loop
    all_images = []

    for i in range(25):
        real_image, real_target = real_iter.next()
        real_image = convert_to_tensor_to_im(real_image.numpy())
        all_images.append((real_image, "real"))
        im = fake_samples[i].permute(1, 2, 0).numpy().squeeze()*std + mean

        fake_image = convert_to_tensor_to_im(np.float32(im))
        all_images.append((fake_image, "fake"))
    fake_count = 0
    real_count = 0
    correct_fake = 0
    correct_real = 0
    random.shuffle(all_images)
    i = 1
    img, target = all_images[0]
    print('first example')
    print(img, target)

    if target == "fake":
        fake_count += 1
    if target == "real":
        real_count += 1
    layout = [[sg.Button("exit", key='exit')], [sg.Button('fake', key='fake'), sg.Button('real', key='real')], [
        sg.Image(data=img, key='img')]]
    window = sg.Window("Demo", layout, margins=(400, 400))

    while True:
        window.Refresh()
        if fake_count + real_count >= 49:
            break
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == "fake" or event == "real":

            img, target = all_images[i]
            if event == "fake":
                fake_count += 1
                if event == target:
                    correct_fake += 1

            if event == "real":
                real_count += 1
                if event == target:
                    correct_real += 1

            window['img'].update(data=img)
            i += 1
        if event == "exit":
            break
        window.Refresh()

    window.close()
    a = fake_count/50
    b = real_count/50

    print(f'User labelled {correct_fake} out of a total of {25} fake images')
    print(f'User labelled {correct_real} out of a total of {25} real images')
    print(fake_count)
    print(real_count)
