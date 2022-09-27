import numpy as np
from matplotlib import pyplot as plt
import functools

N = 50


@functools.lru_cache(maxsize=1)
def gen_boundary(lightning_coords):

    # Lightning coords have phi = 0
    boundaries = {coord: 0 for coord in lightning_coords}

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


def solve_laplace(lightning_coords):
    A = np.zeros((N, N, N, N), dtype='d')
    b = np.zeros((N, N), dtype='d')

    dx = 1 / (N - 1)

    boundaries = gen_boundary(lightning_coords)

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            if (i, j) in boundaries:
                continue
            A[i, j, i, j] = -4/dx**2
            A[i, j, i - 1, j] = 1/dx**2
            A[i, j, i + 1, j] = 1/dx**2
            A[i, j, i, j - 1] = 1/dx**2
            A[i, j, i, j + 1] = 1/dx**2

    for boundary_coord, phi in boundaries.items():
        i, j = boundary_coord
        A[i, j, i, j] = 1
        b[i, j] = phi

    return np.linalg.tensorsolve(A, b)
    # return cupy.linalg.tensorsolve(cupy.asarray(A), cupy.asarray(b))


def main():
    lightning_coords = frozenset([(N//2, N//2)])
    grid = solve_laplace(lightning_coords)
    # render grid
    plt.imshow(grid, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
