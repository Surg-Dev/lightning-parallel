import numpy as np
from matplotlib import pyplot as plt
import functools
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

N = 100


@functools.cache
def make_initial_boundaries():
    boundaries = {}

    # Circle around center is ground, has phi = 1
    t = np.linspace(0, 2*np.pi, int(2 * np.pi * (N / 2)))
    ys = N//2 + np.round(0.8 * N / 2 * np.sin(t)).astype(int)
    xs = N//2 + np.round(0.8 * N / 2 * np.cos(t)).astype(int)

    for y, x in zip(ys, xs):
        boundaries[(y, x)] = 1

    # Border always has phi = 0
    for i in range(N):
        boundaries[(0, i)] = 0
        boundaries[(N-1, i)] = 0
        boundaries[(i, 0)] = 0
        boundaries[(i, N-1)] = 0

    return boundaries


def gen_boundaries(lightning_coords):
    # Lightning coords have phi = 0
    return make_initial_boundaries() | {coord: 0 for coord in lightning_coords}


def gen_problem(lightning_coords):

    A = np.zeros((N * N, N * N))
    b = np.zeros((N * N))

    def coord(i, j): return i * N + j

    dx = 1 / (N - 1)

    boundaries = gen_boundaries(lightning_coords)

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            if (i, j) in boundaries:
                continue

            A[coord(i, j), coord(i, j)] = -4/dx**2
            A[coord(i, j), coord(i-1, j)] = 1/dx**2
            A[coord(i, j), coord(i+1, j)] = 1/dx**2
            A[coord(i, j), coord(i, j-1)] = 1/dx**2
            A[coord(i, j), coord(i, j+1)] = 1/dx**2

    for boundary_coord, phi in boundaries.items():
        i, j = boundary_coord
        A[coord(i, j), coord(i, j)] = 1
        b[coord(i, j)] = phi

    return csr_matrix(A), b


def solve_problem(lightning_coords):
    A, b = gen_problem(lightning_coords)
    # print('holy shit')
    # return np.linalg.tensorsolve(A, b).reshape((N, N))
    return spsolve(A, b).reshape((N, N))


def find_adj(coords):
    '''Find adjacent pixels to each pixel in coords'''
    adj = set()
    for y, x in coords:
        if y > 0:
            adj.add((y-1, x))
        if y < N - 1:
            adj.add((y+1, x))
        if x > 0:
            adj.add((y, x-1))
        if x < N - 1:
            adj.add((y, x+1))
    return list(adj - set(coords))


def choose_next_lightning(grid, adj, eta=2):
    probabilities = np.array([grid[y, x] ** eta for y, x in adj])
    probabilities /= probabilities.sum()
    new_pixel = adj[np.random.choice(len(adj),
                                     p=probabilities)]
    return new_pixel


def display_lightning(lightning_coords):
    display = np.zeros((N, N))
    for y, x in lightning_coords:
        display[y, x] = 1

    plt.imshow(display)
    plt.show()


def main():
    lightning_coords = [(N//2, N//2)]

    while True:
        grid = solve_problem(lightning_coords)
        possible_next_lightning = find_adj(lightning_coords)
        next_lightning = choose_next_lightning(grid, possible_next_lightning)

        if grid[next_lightning] == 1:
            break

        lightning_coords.append(next_lightning)

    display_lightning(lightning_coords)


if __name__ == '__main__':
    main()
