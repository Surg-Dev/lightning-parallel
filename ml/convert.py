import os
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

    import sys

    if len(sys.argv) != 2:
        print('Usage: python convert.py <folder>')
        exit(1)

    data_folder = sys.argv[1]

    dataset = []
    for folder in tqdm(os.listdir(data_folder)):
        all_files = os.listdir(os.path.join(data_folder, folder))
        # choose at most 5 random files
        files = np.random.choice(all_files, min(
            5, len(all_files)), replace=False)
        for file in files:
            dataset.append(convert_problem(pickle.load(
                open(os.path.join(data_folder, folder, file), 'rb'))))

    torch.save(dataset, 'dataset.pt')
