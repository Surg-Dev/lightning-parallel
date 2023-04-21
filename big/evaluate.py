import sys
from test import NET_BOLT, NET_CANDIDATE, NET_EMPTY, NET_GROUND, NET_WALL, draw_map

import matplotlib.pyplot as plt
import numpy as np
import poisson_disc as pd
import torch

from make_data import EMPTY, END, GROUND, START, WALL, N
from train import build_unet
import time


def load_ppm(filename):
    with open(filename, "rb") as f:
        assert f.readline().strip() == b"P6"
        assert f.readline().strip().startswith(b"#")
        width, height = [int(x) for x in f.readline().strip().split()]
        assert (width, height) == (N, N)
        assert f.readline().strip() == b"255"
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(height, width, 3)

        map = np.full((N, N), EMPTY, dtype=np.uint8)
        for y in range(N):
            for x in range(N):
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


def update_candidates(candidates, map, y, x):
    good = lambda type: type == NET_EMPTY or type == NET_CANDIDATE or type == NET_GROUND

    if x > 0 and good(map[y, x - 1]):
        candidates.add((y, x - 1))
    if x < N - 1 and good(map[y, x + 1]):
        candidates.add((y, x + 1))
    if y > 0 and good(map[y - 1, x]):
        candidates.add((y - 1, x))
    if y < N - 1 and good(map[y + 1, x]):
        candidates.add((y + 1, x))
    if x > 0 and y > 0 and good(map[y - 1, x - 1]):
        candidates.add((y - 1, x - 1))
    if x < N - 1 and y > 0 and good(map[y - 1, x + 1]):
        candidates.add((y - 1, x + 1))
    if x > 0 and y < N - 1 and good(map[y + 1, x - 1]):
        candidates.add((y + 1, x - 1))
    if x < N - 1 and y < N - 1 and good(map[y + 1, x + 1]):
        candidates.add((y + 1, x + 1))


class Node:
    def __init__(self, point, parent) -> None:
        self.point = point
        self.parent = parent
        self.children = []
        self.is_lead = False
        self.depth = 0


class Dag:
    def __init__(self, map, start) -> None:
        self.map = map
        self.start = start
        self.nodes = {start: Node(start, None)}

    def add_point(self, point):
        neighbors = self._get_neighbors(point[0], point[1])
        parent = neighbors[0]

        self.nodes[parent].children.append(point)
        self.nodes[point] = Node(point, parent)

    def _get_neighbors(self, y, x):
        neighbors = []
        if x > 0 and self.map[y, x - 1] == NET_BOLT:
            neighbors.append((y, x - 1))
        if x < N - 1 and self.map[y, x + 1] == NET_BOLT:
            neighbors.append((y, x + 1))
        if y > 0 and self.map[y - 1, x] == NET_BOLT:
            neighbors.append((y - 1, x))
        if y < N - 1 and self.map[y + 1, x] == NET_BOLT:
            neighbors.append((y + 1, x))
        if x > 0 and y > 0 and self.map[y - 1, x - 1] == NET_BOLT:
            neighbors.append((y - 1, x - 1))
        if x < N - 1 and y > 0 and self.map[y - 1, x + 1] == NET_BOLT:
            neighbors.append((y - 1, x + 1))
        if x > 0 and y < N - 1 and self.map[y + 1, x - 1] == NET_BOLT:
            neighbors.append((y + 1, x - 1))
        if x < N - 1 and y < N - 1 and self.map[y + 1, x + 1] == NET_BOLT:
            neighbors.append((y + 1, x + 1))
        return neighbors

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
                stdDev = -(max_depth * max_depth) / (np.log(0.35) * 2.0)

                eTerm = -end_node.depth * end_node.depth
                eTerm /= 2.0 * stdDev
                eTerm = np.exp(eTerm) * 0.5
                result[end_node.point] = eTerm

            self._build_intensity(result, end_node)

    def draw(self, end):
        node = self.nodes[end]
        while node:
            node.is_lead = True

            for child in node.children:
                child = self.nodes[child]
                if not child.is_lead:
                    self._build_branch(child)

            node = self.nodes.get(node.parent, None)

        result = np.zeros((N, N), dtype=np.float32)
        self._build_intensity(result, self.nodes[self.start])
        return result


def simulate(model, map, start, end, device, eta=1):
    candidates = set()
    update_candidates(candidates, map, start[0], start[1])

    points = pd.Bridson_sampling(radius=0.1)
    for point in points:
        y, x = int(point[0] * N), int(point[1] * N)
        if map[y, x] == NET_EMPTY:
            map[y, x] = NET_GROUND

    dag = Dag(map, start)

    while candidates:
        x = torch.zeros((1, 1, N, N), device=device)
        x[0, 0, map == NET_BOLT] = -1
        x[0, 0, map == NET_GROUND] = 1
        y = model(x) ** eta
        y = y[0, 0].detach().cpu().numpy()

        # choose out of candidates with probability given by y
        candidate_list = list(candidates)
        probs = []
        for candidate in candidate_list:
            probs.append(y[candidate[0], candidate[1]])

        probs = np.array(probs)
        probs[probs < 0] = 0
        probs[np.isnan(probs)] = 0
        probs /= probs.sum()
        chosen = np.random.choice(len(candidate_list), p=probs)
        chosen = candidate_list[chosen]

        # update map
        map[chosen[0], chosen[1]] = NET_BOLT
        dag.add_point(chosen)
        candidates.remove(chosen)
        update_candidates(candidates, map, chosen[0], chosen[1])

        last_added = chosen

        if end in candidates:
            break

    intensities = dag.draw(last_added)
    return map, intensities


def save_intensities(intensities, file):
    with open(file, "wb") as f:
        f.write(intensities.tobytes())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    if len(sys.argv) != 3:
        print("Usage: python3 evaluator.py <model.pt> <ppm>")
        sys.exit(1)

    model = build_unet()
    model.load_state_dict(torch.load(sys.argv[1]))
    model.eval()
    model.to(device)

    map, start, end = load_ppm(sys.argv[2])
    map, intensities = simulate(model, map, start, end, device)
    save_intensities(intensities, "out.bin")

    # with open("foo.bin", "rb") as f:
    #     data = np.fromfile(f, dtype=np.float32).reshape((N, N))
    #     data /= data.max()
    #     plt.imshow(data, cmap="gray", vmin=0, vmax=1)
    #     plt.show()
