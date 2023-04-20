import os
import pickle
import random
import uuid

from lightning_sparse import *


def make_initial_boundaries():
    boundaries = {}

    num_ground = random.randint(1, 10)

    for _ in range(num_ground):
        square_width = random.randint(1, 20)
        y = random.randint(1, N - square_width - 1)
        x = random.randint(1, N - square_width - 1)
        for i in range(y, y + square_width):
            for j in range(x, x + square_width):
                boundaries[(i, j)] = 1

    return boundaries


def make_initial_lightning(boundaries):
    while True:
        y = random.randint(1, N - 2)
        x = random.randint(1, N - 2)
        if (y, x) not in boundaries:
            return [(y, x)]


def save_grid(folder, grid, name, i, boundaries, lightning_coords, adj):
    folder = f'{folder}/{name}'
    os.makedirs(folder, exist_ok=True)

    with open(f'{folder}/{i}.pickle', 'wb') as f:
        pickle.dump({
            'grid': grid,
            'boundaries': boundaries,
            'lightning_coords': lightning_coords,
            'adj': adj
        }, f)


if __name__ == '__main__':

    import sys

    if len(sys.argv) != 2:
        print('Usage: python make_data.py <folder>')
        exit(1)

    folder = sys.argv[1]

    iteration = 0
    while True:
        print(f'Iteration {iteration}')
        boundaries = make_initial_boundaries()
        lightning_coords = make_initial_lightning(boundaries)

        name = str(uuid.uuid4())
        i = 0
        while True:
            grid = solve_problem(lightning_coords, boundaries)
            possible_next_lightning = find_adj(lightning_coords)

            save_grid(folder, grid, name, i, boundaries, lightning_coords,
                      possible_next_lightning)

            next_lightning = choose_next_lightning(
                grid, possible_next_lightning)

            if grid[next_lightning] == 1:
                break

            lightning_coords.append(next_lightning)
            i += 1

        # display_lightning(lightning_coords, boundaries)
        iteration += 1
