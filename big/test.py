import os
import torch
import numpy as np
import matplotlib.pyplot as plt

for file in os.listdir("data"):
    print(file)
    dataset = torch.load(os.path.join("data", file))
    print(len(dataset))

    for i, (bolt, candidate) in enumerate(dataset):
        print(bolt.shape, candidate.shape)
        assert bolt.shape == candidate.shape
        # make rgb image and render with matplotlib
        candidate /= candidate.sum()

        rgb = np.zeros((bolt.shape[0], bolt.shape[1], 3), dtype=np.float32)
        rgb[bolt == 1] = [1, 1, 1]
        rgb[candidate > 0] = [1, 0, 0]

        plt.imshow(rgb)
        plt.show()
