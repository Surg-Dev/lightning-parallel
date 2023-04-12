import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


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

        self.up = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(1, 16)
        self.e2 = encoder_block(16, 32)
        self.e3 = encoder_block(32, 64)

        """ Bottleneck """
        self.b = conv_block(64, 128)

        """ Decoder """
        self.d1 = decoder_block(128, 64)
        self.d2 = decoder_block(64, 32)
        self.d3 = decoder_block(32, 16)

        """ Classifier """
        self.outputs = nn.Conv2d(16, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
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


def custom_loss(y, y_hat, mask):
    mask = y_hat > 0
    return nn.MSELoss()(y_hat * mask, y * mask)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = torch.load('data/dataset.pt')
    model = build_unet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(dataset, batch_size=16,
                            shuffle=True, num_workers=4, pin_memory=True)

    losses = []

    for epoch in range(10):
        print(f'Epoch {epoch}')

        for _, (x, (y, mask)) in enumerate(tqdm(dataloader)):
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            y_hat = model(x)
            loss = custom_loss(y, y_hat, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.show()

    torch.save(model, 'data/model.pt')


if __name__ == "__main__":
    # inputs = torch.randn((2, 1, 64, 64))
    # model = build_unet()
    # y = model(inputs)
    # print(y.shape)
    train()
