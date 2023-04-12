import glob
import pickle

import numpy as np
import torch
from lightning_sparse import N
from tqdm import tqdm


def make_input(boundaries, lightning_coords):
    input = np.zeros((N, N), dtype=np.float32)
    for (y, x) in boundaries:
        input[y, x] = 1
    for (y, x) in lightning_coords:
        input[y, x] = -1
    return input.reshape((1, N, N))


def make_output(grid, adj):
    mask = np.zeros((N, N), dtype=np.float32)
    for (y, x) in adj:
        mask[y, x] = 1

    grid *= mask
    grid /= np.sum(grid)

    return grid.astype(np.float32).reshape((1, N, N)), mask.reshape((1, N, N))


def convert_problem(problem):

    grid = problem['grid']
    boundaries = problem['boundaries']
    lightning_coords = problem['lightning_coords']
    adj = problem['adj']

    return make_input(boundaries, lightning_coords), make_output(grid, adj)


if __name__ == '__main__':
    files = glob.glob('data/*.pickle')
    dataset = []
    for i in tqdm(range(25000)):
        file = files[i]
        with open(file, 'rb') as f:
            problem = pickle.load(f)
            dataset.append(convert_problem(problem))

    torch.save(dataset, 'data/dataset.pt')
