import sys
from test import NET_BOLT, NET_GROUND, NET_WALL, NET_ATTRACTOR, draw_map

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate import Bolt, add_noise, load_model, load_ppm

import numpy as np
from scipy import signal


class TiledGrid:
    def __init__(self, map, start, end, tile_size, overlap, pad=0) -> None:
        self.original_map_height, self.original_map_width = map.shape

        self.tile_size = tile_size
        self.overlap = overlap
        self.tile_height = self._calculate_side(self.original_map_height)
        self.tile_width = self._calculate_side(self.original_map_width)

        map_height = (self.tile_height - 1) * (
            self.tile_size - self.overlap
        ) + self.tile_size
        map_width = (self.tile_width - 1) * (
            self.tile_size - self.overlap
        ) + self.tile_size

        print(self.tile_width, self.tile_height)
        print(map_width, map_height)

        self.map = np.full((map_height, map_width), NET_WALL, dtype=np.uint8)
        self.map[: self.original_map_height, : self.original_map_width] = map
        add_noise(self.map, radius=20.4 / min(map.shape))
        self.bolt = Bolt(self.map, start, end)
        self.pad = pad

        self.gaussian = signal.gaussian(self.tile_size, self.tile_size // 8)
        self.gaussian = np.outer(self.gaussian, self.gaussian) ** 2

    def simulate(self, model, device, eta=1):
        i = 0
        while not self.bolt.is_complete():
            x = self._make_x()
            poisson = self._make_poisson(x, model, device)
            self.bolt.add_new(self.map, poisson, eta)

            if i % 2000 == 0:
                rgb = draw_map(self.map)
                plt.imshow(rgb)
                plt.show()

                plt.imshow(poisson, cmap="gray")
                plt.show()

            i += 1

        return self.map[: self.original_map_height, : self.original_map_width]

    def _calculate_side(self, length):
        # find smallest n s.t. (n + 1) * (tile_size - overlap) + tile_size >= length
        return int(
            np.ceil((length - self.tile_size) / (self.tile_size - self.overlap) + 1)
        )

    def _make_x(self):
        num_tiles = self.tile_height * self.tile_width
        batch = torch.zeros(
            (num_tiles, 1, self.tile_size + 2 * self.pad, self.tile_size + 2 * self.pad)
        )
        for i in range(self.tile_height):
            for j in range(self.tile_width):
                start_y = int(i * (self.tile_size - self.overlap))
                start_x = int(j * (self.tile_size - self.overlap))
                map_view = self.map[
                    start_y : start_y + self.tile_size,
                    start_x : start_x + self.tile_size,
                ]
                tile = batch[
                    i * self.tile_width + j,
                    0,
                    self.pad : self.pad + self.tile_size,
                    self.pad : self.pad + self.tile_size,
                ]
                tile[map_view == NET_BOLT] = -1
                tile[map_view == NET_GROUND] = 1
                tile[map_view == NET_ATTRACTOR] = 1
        return batch

    def _make_poisson(self, x, model, device):
        x = x.to(device)
        y = model(x).cpu().detach().numpy()
        poisson = np.zeros_like(self.map, dtype=np.float32)
        weights = np.zeros_like(self.map, dtype=np.float32)

        for i in range(self.tile_height):
            for j in range(self.tile_width):
                start_y = int(i * (self.tile_size - self.overlap))
                start_x = int(j * (self.tile_size - self.overlap))
                poisson_view = poisson[
                    start_y : start_y + self.tile_size,
                    start_x : start_x + self.tile_size,
                ]
                weight_view = weights[
                    start_y : start_y + self.tile_size,
                    start_x : start_x + self.tile_size,
                ]
                tile = y[
                    i * self.tile_width + j,
                    0,
                    self.pad : self.pad + self.tile_size,
                    self.pad : self.pad + self.tile_size,
                ]
                # poisson_view[:] = poisson_view + tile * self.gaussian
                poisson_view[:] = poisson_view + tile
                weight_view[:] = weight_view + self.gaussian

        normalized = poisson / weights
        # return normalized
        return poisson


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) != 3:
        print("Usage: python3 tile.py <model.pt> <ppm>")
        sys.exit(1)

    model = load_model(sys.argv[1], device)
    map, start, end = load_ppm(sys.argv[2])
    grid = TiledGrid(map, start, end, 512, 0)
    # grid = TiledGrid(map, start, end, 256, 64)
    map, intensities = grid.simulate(model, device)
    rgb = draw_map(map)
    plt.imshow(rgb)
    plt.show()


if __name__ == "__main__":
    main()
