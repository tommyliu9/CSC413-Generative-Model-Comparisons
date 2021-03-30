import torch.utils
import torch
import torch.utils.data
import warnings
import random
import sklearn.preprocessing
import norbert
import pandas as pd
import musdb

import torchaudio
from tqdm import tnrange, tqdm_notebook, tqdm
class SimpleMUSDBDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        subset='train',
        split='train', 
        target='vocals',
        seq_duration=None,
    ):
        """MUSDB18 Dataset wrapper
        """
        self.seq_duration = seq_duration
        self.target = target
        self.mus = musdb.DB(
            download=True,
            split=split,
            subsets=subset,
        )

    def __getitem__(self, index):
        track = self.mus[index]
        track.chunk_start = random.uniform(0, track.duration - self.seq_duration)
        track.chunk_duration = self.seq_duration
        x = track.audio.T
        y = track.targets[self.target].audio.T
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


    def __len__(self):
        return len(self.mus)
    
class FullMUSDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        subset='train',
        split='train',
        target='vocals',
        seq_duration=None,
    ):
        """MUSDB18 Dataset wrapper
        """
        self.seq_duration = seq_duration
        self.target = target

        self.mus = musdb.DB(root="./data/",
            split=split,
            subsets=subset,
        )

    def __getitem__(self, index):
        track = self.mus[index]
        track.chunk_start = random.uniform(0, track.duration - self.seq_duration)
        track.chunk_duration = self.seq_duration
        x = track.audio.T
        y = track.targets[self.target].audio.T
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.mus)

if __name__ == "__main__":
    import torch.nn as nn

    train_dataset = FullMUSDBDataset(seq_duration=5.0)
    train_sampler = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    torchaudio.transforms.Spectrogram()
    stft = torch.stft()
    spec = torch.transforms.Spectrogram()
    transform = nn.Sequential(stft, spec)

    print(len(train_dataset))

    print(len(train_sampler))
    x,y = train_datset[7]
    X = transform(x[None])
    Y = transform(y[None])
    
    