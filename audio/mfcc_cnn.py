import io
import os
import wave

import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(51136, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x,dim=1)


def save_mfcc(waveform, sample_rate):
    mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)

    plt.figure()
    fig1 = plt.gcf()
    plt.imshow(mfcc_spectrogram.log2()[0, :, :].numpy(), cmap='viridis')
    plt.draw()

    fig1.savefig('audio/test/test/test.png', dpi=100)

def make_dataloader(data_path):
    yes_no_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([transforms.Resize((201,81)),
                                      transforms.ToTensor()
                                      ])
    )

    test_dataloader = torch.utils.data.DataLoader(
        yes_no_dataset,
        batch_size=15,
        num_workers=0,
        shuffle=True
    )

    return test_dataloader


def mfcc_spectrogram_prediction(wave_file_path, models_path):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = CNNet().to(device)
    model = torch.load(models_path)

    model.eval()

    waveform, sample_rate = torchaudio.load(wave_file_path)
    save_mfcc(waveform, sample_rate)

    dataloader = make_dataloader('audio/test')

    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        output = model(X)

    return output
