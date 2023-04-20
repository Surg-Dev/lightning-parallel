import os
import sys
from test import NET_BOLT, NET_CANDIDATE, NET_EMPTY, NET_GROUND, NET_WALL

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from make_data import N

# based off of https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(1, 32)
        self.e2 = encoder_block(32, 64)
        self.e3 = encoder_block(64, 128)

        """ Bottleneck """
        self.b = conv_block(128, 256)

        """ Decoder """
        self.d1 = decoder_block(256, 128)
        self.d2 = decoder_block(128, 64)
        self.d3 = decoder_block(64, 32)

        """ Classifier """
        self.outputs = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """Encoder"""
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        """ Bottleneck """
        b = self.b(p3)

        """ Decoder """
        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        """ Classifier """
        outputs = self.outputs(d3)

        return outputs


def custom_loss(y, y_hat, map):
    mask = map == NET_CANDIDATE
    return nn.MSELoss()(y_hat * mask, y * mask) * (N * N) / mask.sum()


def train(model, dataset, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainloader = DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )

    losses = []
    for map, y in tqdm(trainloader):
        map = map.to(device).reshape(-1, 1, N, N)
        y = y.to(device).reshape(-1, 1, N, N)
        x = torch.zeros_like(y, device=device)
        x[map == NET_BOLT] = -1
        x[map == NET_GROUND] = 1
        y_hat = model(x)
        loss = custom_loss(y, y_hat, map)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) == 1:
        print("Usage: python train.py dataset_folder modelout [modelin]")
        exit(1)

    dataset_folder = sys.argv[1]
    model_out = sys.argv[2]

    if len(sys.argv) == 4:
        model = build_unet()
        model.load_state_dict(torch.load(sys.argv[3]))
    else:
        model = build_unet()

    model.to(device)

    for dataset in os.listdir(dataset_folder):
        dataset = os.path.join(dataset_folder, dataset)
        dataset = torch.load(dataset)
        train(model, dataset, device)
        torch.save(model.state_dict(), model_out)
