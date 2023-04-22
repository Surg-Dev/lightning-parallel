import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys


NET_EMPTY = 0
NET_BOLT = 1
NET_CANDIDATE = 2
NET_WALL = 3
NET_GROUND = 4
NET_ATTRACTOR = 5


def draw_map(map):
    rgb = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.float32)
    rgb[map == NET_BOLT] = [1, 1, 1]
    rgb[map == NET_CANDIDATE] = [1, 0, 0]
    rgb[map == NET_WALL] = [0, 1, 0]
    rgb[map == NET_GROUND] = [0, 0, 1]
    rgb[map == NET_ATTRACTOR] = [1, 1, 0]

    return rgb


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <data folder>")
        sys.exit(1)

    folder = sys.argv[1]
    for file in os.listdir(folder):
        print(file)
        dataset = torch.load(os.path.join(folder, file))
        print(len(dataset))

        for i, (map, candidate) in enumerate(dataset):
            print(map.shape, candidate.shape)
            assert map.shape == candidate.shape
            # make rgb image and render with matplotlib
            candidate /= candidate.sum()

            # show map and candidate side by side
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(draw_map(map))
            ax2.imshow(candidate, cmap="gray")
            plt.show()
