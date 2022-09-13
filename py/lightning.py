import numpy as np
import tqdm
import matplotlib.pyplot as plt


def write_boundary(x, lightning_coords):
    '''Write 1 if ground, and 0 if lightning'''
    for coord in lightning_coords:
        x[coord] = 0

    # parameterize a circle around the center
    w, h = x.shape
    t = np.linspace(0, 2*np.pi, int(2 * np.pi * (w / 2)))
    x[w//2 + np.round(0.8 * w / 2 * np.cos(t)).astype(int),
      h//2 + np.round(0.8 * h / 2 * np.sin(t)).astype(int)] = 1


# http://www.vallis.org/salon2/lecture2-script.html


def jacobian(grid):
    newgrid = np.zeros(shape=grid.shape, dtype=grid.dtype)

    # apply evolution operator
    newgrid[1:-1, 1:-1] = 0.25 * (grid[1:-1, :-2] + grid[1:-1, 2:] +
                                  grid[:-2, 1:-1] + grid[2:, 1:-1])

    return newgrid


def find_adj(coords, grid_shape):
    '''Find adjacent pixels to each pixel in coords'''
    adj = set()
    for y, x in coords:
        if y > 0:
            adj.add((y-1, x))
        if y < grid_shape[0] - 1:
            adj.add((y+1, x))
        if x > 0:
            adj.add((y, x-1))
        if x < grid_shape[1] - 1:
            adj.add((y, x+1))
    return list(adj - set(coords))


def main():
    grid_size = 100
    lightning_coords = [(grid_size//2, grid_size//2)]

    eta = 2

    while True:
        grid = np.random.randn(grid_size, grid_size)
        write_boundary(grid, lightning_coords)

        # solve laplace equation
        for _ in tqdm.tqdm(range(10000)):
            grid = jacobian(grid)
            write_boundary(grid, lightning_coords)

        # choose next growth site
        adj = find_adj(lightning_coords, grid.shape)
        probabilities = np.array([grid[y, x] ** eta for y, x in adj])
        probabilities /= probabilities.sum()
        new_pixel = adj[np.random.choice(len(adj),
                                         p=probabilities)]

        # hit the ground
        if grid[new_pixel] == 1:
            break

        lightning_coords.append(new_pixel)

    # plot lightning
    grid[grid != 0] = 1
    grid = 1 - grid

    plt.imshow(grid, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
