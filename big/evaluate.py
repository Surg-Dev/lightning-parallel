import os
import random
import subprocess
import sys
import tempfile
from test import (
    NET_BOLT,
    NET_CANDIDATE,
    NET_EMPTY,
    NET_GROUND,
    NET_WALL,
    NET_ATTRACTOR,
    draw_map,
)

import matplotlib.pyplot as plt
import numpy as np
import poisson_disc as pd
import torch

from make_data import EMPTY, END, GROUND, START, WALL
from train import build_unet


def load_ppm(filename):
    with open(filename, "rb") as f:
        assert f.readline().strip() == b"P6"
        assert f.readline().strip().startswith(b"#")
        width, height = [int(x) for x in f.readline().strip().split()]
        assert f.readline().strip() == b"255"
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(height, width, 3)

        map = np.full((height, width), EMPTY, dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                val = img[y, x, 0] << 16 | img[y, x, 1] << 8 | img[y, x, 2]
                if val == START:
                    start = (y, x)
                    map[y, x] = NET_BOLT
                elif val == END:
                    end = (y, x)
                    map[y, x] = NET_GROUND
                elif val == GROUND:
                    map[y, x] = NET_GROUND
                elif val == WALL:
                    map[y, x] = NET_WALL
                else:
                    map[y, x] = NET_EMPTY

    return map, start, end


class Node:
    def __init__(self, point, parent) -> None:
        self.point = point
        self.parent = parent
        self.children = []
        self.is_lead = False
        self.depth = 0


def get_neighbors(point, h, w):
    y, x = point
    neighbors = []
    if x > 0:
        neighbors.append((y, x - 1))
    if x < w - 1:
        neighbors.append((y, x + 1))
    if y > 0:
        neighbors.append((y - 1, x))
    if y < h - 1:
        neighbors.append((y + 1, x))
    if x > 0 and y > 0:
        neighbors.append((y - 1, x - 1))
    if x < w - 1 and y > 0:
        neighbors.append((y - 1, x + 1))
    if x > 0 and y < h - 1:
        neighbors.append((y + 1, x - 1))
    if x < w - 1 and y < h - 1:
        neighbors.append((y + 1, x + 1))
    return neighbors


class Dag:
    def __init__(self, start) -> None:
        self.start = start
        self.nodes = {start: Node(start, None)}

    def add_point(self, map, point):
        neighbors = [
            n
            for n in get_neighbors(point, map.shape[0], map.shape[1])
            if map[n] == NET_BOLT
        ]
        parent = random.choice(neighbors)

        self.nodes[parent].children.append(point)
        self.nodes[point] = Node(point, parent)

    def _build_branch(self, node, depth=1):
        node.depth = depth

        for child in node.children:
            child = self.nodes[child]
            if not child.is_lead:
                self._build_branch(child, depth + 1)

    def _find_deepest(self, node):
        if not node.children:
            return node

        max_child = None
        max_depth = 0
        for child in node.children:
            child = self.nodes[child]
            if child.depth > max_depth:
                max_child = child
                max_depth = child.depth

        return self._find_deepest(max_child)

    def _build_intensity(self, result, node):
        for end_node in node.children:
            end_node = self.nodes[end_node]
            if node.is_lead:
                result[end_node.point] = 0.75
            else:
                max_depth = self._find_deepest(end_node).depth
                stdDev = -(max_depth * max_depth) / (np.log(0.3) * 2.0)

                eTerm = -end_node.depth * end_node.depth
                eTerm /= 2.0 * stdDev
                eTerm = np.exp(eTerm) * 0.5
                result[end_node.point] = eTerm

            self._build_intensity(result, end_node)

    def draw(self, end, h, w):
        node = self.nodes[end]
        while node:
            node.is_lead = True

            for child in node.children:
                child = self.nodes[child]
                if not child.is_lead:
                    self._build_branch(child)

            node = self.nodes.get(node.parent, None)

        result = np.zeros((h, w), dtype=np.float32)
        self._build_intensity(result, self.nodes[self.start])
        return result


class Bolt:
    def __init__(self, map, start, end) -> None:
        self.start = start
        self.end = end
        self.dag = Dag(start)
        self.candidates = set()
        self.complete = False

        map[start] = NET_BOLT
        self._update_candidates(map, start)

    def _add_pixel(self, map, pixel):
        map[pixel] = NET_BOLT
        self._update_candidates(map, pixel)
        self.dag.add_point(map, pixel)

    def _update_candidates(self, map, pixel):
        good = (
            lambda type: type == NET_EMPTY
            or type == NET_CANDIDATE
            # eat ground
            or type == NET_GROUND
            or type == NET_ATTRACTOR
        )

        self.candidates.update(
            [
                c
                for c in get_neighbors(pixel, map.shape[0], map.shape[1])
                if good(map[c])
            ]
        )

    def add_new(self, map, poisson, eta):
        poisson **= eta
        ordered_candidates = list(self.candidates)
        probs = np.array([poisson[c] for c in ordered_candidates])
        probs = probs[np.isfinite(probs)]
        probs[probs < 0] = 0
        s = probs.sum()
        probs /= s
        chosen_index = np.random.choice(len(ordered_candidates), p=probs)
        chosen_candidate = ordered_candidates[chosen_index]
        self.candidates.remove(chosen_candidate)
        self._add_pixel(map, chosen_candidate)

        if chosen_candidate == self.end:
            self.complete = True

    def get_intensities(self, h, w):
        if not self.is_complete():
            raise Exception("Bolt is not complete")
        return self.dag.draw(self.end, h, w)

    def is_complete(self):
        return self.complete


def map_to_x(map, device):
    x = torch.zeros((1, 1, map.shape[0], map.shape[1]), device=device)
    x[0, 0, map == NET_BOLT] = -1
    x[0, 0, map == NET_GROUND] = 1
    x[0, 0, map == NET_ATTRACTOR] = 0.5
    return x


def add_noise(map, radius=0.08):
    points = pd.Bridson_sampling(radius=radius)
    for point in points:
        y, x = int(point[0] * map.shape[0]), int(point[1] * map.shape[1])
        if map[y, x] == NET_EMPTY:
            map[y, x] = NET_ATTRACTOR


def simulate(model, map, start, end, device, eta=1):
    add_noise(map, radius=0.08)
    bolt = Bolt(map, start, end)
    i = 0
    while not bolt.is_complete():
        x = map_to_x(map, device)
        poisson = model(x)[0, 0].detach().cpu().numpy()
        bolt.add_new(map, poisson, eta)

        if i % 100 == 0:
            from test import draw_map

            plt.imshow(draw_map(map))
            plt.show()

        i += 1

    intensities = bolt.get_intensities()
    return map, intensities


def convolve(intensities):
    temp_file = tempfile.mktemp()
    with open(temp_file, "wb") as f:
        f.write(intensities.tobytes())

    subprocess.run(["./convolve", temp_file, temp_file])

    with open(temp_file, "rb") as f:
        data = np.fromfile(f, dtype=np.float32).reshape((256, 256))

    os.remove(temp_file)
    return data


def load_model(filename, device):
    model = build_unet()
    model.load_state_dict(torch.load(filename))
    model.eval()
    model.to(device)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) != 3:
        print("Usage: python3 evaluator.py <model.pt> <ppm>")
        sys.exit(1)

    model = load_model(sys.argv[1], device)
    map, start, end = load_ppm(sys.argv[2])
    map, intensities = simulate(model, map, start, end, device)
    intensities = convolve(intensities)

    plt.imshow(intensities, cmap="gray")
    plt.show()
