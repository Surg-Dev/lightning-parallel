import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
from lightning_sparse import N

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
    mask = y > 0
    return nn.MSELoss()(y_hat * mask, y * mask) * (N * N) / mask.sum()
    # return nn.CrossEntropyLoss()(y_hat * mask, y * mask) / mask.sum()


def simulate_one(model, x, y, mask):
    y_hat = model(x) * mask

    import matplotlib.pyplot as plt
    plt.imshow(y_hat[0].cpu().detach().numpy().flatten().reshape(N, N))
    plt.show()
    plt.imshow(y[0].cpu().detach().numpy().flatten().reshape(N, N))
    plt.show()

    eta = 3
    y_hat = (y_hat * mask) ** eta / mask.sum()
    import numpy as np

    next_choice_by_probability_distribution = np.random.choice(
        np.arange(N * N), p=y_hat[0].cpu().detach().numpy().flatten())
    next_choice = np.unravel_index(
        next_choice_by_probability_distribution, (N, N))


def simulate_random():
    from make_data import make_initial_boundaries, make_initial_lightning, display_lightning
    from lightning_sparse import solve_problem, find_adj, choose_next_lightning

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet().to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    boundaries = make_initial_boundaries()
    lightning_coords = make_initial_lightning(boundaries)

    import time
    go = True
    while go:
        # display_lightning(lightning_coords, boundaries)

        x = np.zeros((N, N))
        for lightning in lightning_coords:
            x[lightning] = -1

        for boundary in boundaries:
            x[boundary] = 1

        x = torch.from_numpy(x).float().reshape(1, 1, N, N).to(device)
        import time
        start = time.time()
        grid = model(x).cpu().detach().numpy().reshape(N, N)
        end = time.time()
        print(end - start)
        grid[grid < 0] = 0
        # import matplotlib.pyplot as plt
        # plt.imshow(grid)
        # plt.show()

        # grid = solve_problem(lightning_coords, boundaries)
        possible_next_lightning = find_adj(lightning_coords)

        mask = np.zeros((N, N))
        for possible in possible_next_lightning:
            mask[possible] = 1

        grid2 = grid * mask
        # plt.imshow(grid2)
        # plt.show()

        # actual = solve_problem(lightning_coords, boundaries) * mask
        # plt.imshow(actual)
        # plt.show()

        next_lightning = choose_next_lightning(
            grid, possible_next_lightning)
            
        for adj in possible_next_lightning:
            if x[0, 0, adj[0], adj[1]] == 1:
                go = False
                break

        lightning_coords.append(next_lightning)

    print(len(lightning_coords))

    display_lightning(lightning_coords, boundaries)

def simulate_ground():
    from make_data import make_initial_boundaries, make_initial_lightning, display_lightning
    from lightning_sparse import solve_problem, find_adj, choose_next_lightning

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet().to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    # boundaries = make_initial_boundaries()
    # lightning_coords = make_initial_lightning(boundaries)


    boundaries = []
    for i in range(N):
        for j in range(5):
            boundaries.append((N - j - 1, i))

    lightning_coords = [(10, N // 2)]

    import time
    go = True
    while go:
        # display_lightning(lightning_coords, boundaries)

        x = np.zeros((N, N))
        for lightning in lightning_coords:
            x[lightning] = -1

        for boundary in boundaries:
            x[boundary] = 1

        x = torch.from_numpy(x).float().reshape(1, 1, N, N).to(device)
        import time
        start = time.time()
        grid = model(x).cpu().detach().numpy().reshape(N, N)
        end = time.time()
        print(end - start)
        grid[grid < 0] = 0
        # import matplotlib.pyplot as plt
        # plt.imshow(grid)
        # plt.show()

        # grid = solve_problem(lightning_coords, boundaries)
        possible_next_lightning = find_adj(lightning_coords)

        mask = np.zeros((N, N))
        for possible in possible_next_lightning:
            mask[possible] = 1

        grid2 = grid * mask
        # plt.imshow(grid2)
        # plt.show()

        # actual = solve_problem(lightning_coords, boundaries) * mask
        # plt.imshow(actual)
        # plt.show()

        next_lightning = choose_next_lightning(
            grid, possible_next_lightning, eta=3)
            
        for adj in possible_next_lightning:
            if x[0, 0, adj[0], adj[1]] == 1:
                go = False
                break

        lightning_coords.append(next_lightning)

    print(len(lightning_coords))

    display_lightning(lightning_coords, boundaries)

def simulate():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet().to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    dataset = torch.load('originalset.pt')
    val_set, _ = dataset[100:], dataset[:100]

    # test the model
    testloader = DataLoader(val_set, pin_memory=True)
    import matplotlib.pyplot as plt

    for _, (x, (y, mask)) in enumerate(tqdm(testloader)):
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        simulate_one(model, x, y, mask)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = torch.load('dataset.pt')
    # _, train_set = dataset[:100], dataset[100:]
    train_set = dataset

    model = build_unet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainloader = DataLoader(train_set, batch_size=16,
                             shuffle=True, num_workers=4, pin_memory=True)

    losses = []
    for _, (x, (y, mask)) in enumerate(tqdm(trainloader)):
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        y_hat = model(x)
        loss = custom_loss(y, y_hat, mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    torch.save(model.state_dict(), 'model.pt')
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    # inputs = torch.randn((2, 1, 128, 128))
    # model = build_unet()
    # y = model(inputs)
    # print(y.shape)
    # train()
    simulate_ground()
