import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from uuid import uuid4

import numpy as np
import torch
from tqdm import tqdm

BIG_N = 512
SMALL_N = 256

EMPTY = 0x000000
WALL = 0x00FF00
START = 0xFF0000
GROUND = 0x0000FF
END = 0xFFFFFF


def has_problematic_overlap(problem, rect, type):
    x, y, w, h = rect
    for i in range(y, y + h):
        for j in range(x, x + w):
            if problem[i, j] != EMPTY and problem[i, j] != type:
                return True
    return False


def random_rect(problem, w, h, type):
    for _ in range(10000):
        x, y = random.randint(0, BIG_N - w), random.randint(0, BIG_N - h)
        if not has_problematic_overlap(problem, (x, y, w, h), type):
            problem[y : y + h, x : x + w] = type
            break


def make_problem(max_ground, max_wall, max_ground_len, max_wall_len):
    problem = np.full((BIG_N, BIG_N), EMPTY, dtype=np.uint32)
    random_rect(problem, 1, 1, START)
    random_rect(problem, 1, 1, END)

    num_ground = random.randint(0, max_ground)
    num_wall = random.randint(0, max_wall)

    while num_ground > 0 or num_wall > 0:
        if num_ground > 0:
            w, h = random.randint(1, max_ground_len), random.randint(1, max_ground_len)
            random_rect(problem, w, h, GROUND)
            num_ground -= 1

        if num_wall > 0:
            w, h = random.randint(1, max_wall_len), random.randint(1, max_wall_len)
            random_rect(problem, w, h, WALL)
            num_wall -= 1

    return problem


def is_solvable(problem):
    import networkx as nx

    G = nx.Graph()
    for i in range(BIG_N):
        for j in range(BIG_N):
            if problem[i, j] != WALL:
                G.add_node((i, j))

    for i in range(BIG_N):
        for j in range(BIG_N):
            if problem[i, j] != WALL:
                if i + 1 < BIG_N and problem[i + 1, j] != WALL:
                    G.add_edge((i, j), (i + 1, j))
                if j + 1 < BIG_N and problem[i, j + 1] != WALL:
                    G.add_edge((i, j), (i, j + 1))

    start_y, start_x = np.where(problem == START)
    end_y, end_x = np.where(problem == END)

    return nx.has_path(G, (start_y[0], start_x[0]), (end_y[0], end_x[0]))


def make_solvable_problem(max_ground, max_wall, max_ground_len, max_wall_len):
    while True:
        img = make_problem(max_ground, max_wall, max_ground_len, max_wall_len)
        if is_solvable(img):
            return img


def save_problem(problem, filename):
    # save as ppm
    with open(filename, "wb") as f:
        f.write(b"P6\n")
        f.write(f"{BIG_N} {BIG_N}\n".encode("ascii"))
        f.write(b"255\n")
        for row in problem:
            for pixel in row:
                f.write(bytes((pixel >> 16 & 0xFF, pixel >> 8 & 0xFF, pixel & 0xFF)))


def solve_problem(problem):
    input_file = tempfile.mktemp()
    save_problem(problem, input_file)

    output_folder = tempfile.mkdtemp()

    subprocess.run(
        ["../lumosquad-0.1/lumosquad", input_file, f"{output_folder}/out", "1", "0"],
        stdout=subprocess.PIPE,
    )

    files = os.listdir(output_folder)
    files.sort()

    pairs = zip(files[::2], files[1::2])

    def load_bolt(bolt_file):
        result = np.zeros((BIG_N, BIG_N), dtype=np.uint8)
        with open(bolt_file, "rb") as f:
            for i in range(BIG_N):
                for j in range(BIG_N):
                    result[i, j] = f.read(1)[0]
        return result

    def load_candidate(candidate_file):
        result = np.zeros((BIG_N, BIG_N), dtype=np.float32)
        with open(candidate_file, "rb") as f:
            for i in range(BIG_N):
                for j in range(BIG_N):
                    result[i, j] = struct.unpack("f", f.read(4))[0]
        return result

    def get_crop_coords(map):
        while True:
            start_y = random.randint(0, BIG_N - SMALL_N)
            start_x = random.randint(0, BIG_N - SMALL_N)
            view = map[start_y : start_y + SMALL_N, start_x : start_x + SMALL_N]
            if np.any(view == 2):
                return start_y, start_x

    map = np.zeros((BIG_N, BIG_N), dtype=np.uint8)
    for i in range(BIG_N):
        for j in range(BIG_N):
            if problem[i, j] == WALL:
                map[i, j] = 3
            if problem[i, j] == GROUND or problem[i, j] == END:
                map[i, j] = 4

    solutions = []
    for bolt, candidate in pairs:
        # empty = 0, bolt = 1, candidate = 2, wall = 3, ground = 4
        loaded_bolt = load_bolt(os.path.join(output_folder, bolt))
        loaded_candidate = load_candidate(os.path.join(output_folder, candidate))

        map_copy = map.copy()
        map_copy[loaded_bolt == 1] = 1
        map_copy[loaded_bolt == 2] = 2

        # crop em
        start_y, start_x = get_crop_coords(map_copy)
        map_copy = map_copy[start_y : start_y + SMALL_N, start_x : start_x + SMALL_N]
        loaded_candidate = loaded_candidate[
            start_y : start_y + SMALL_N, start_x : start_x + SMALL_N
        ]

        solutions.append((map_copy, loaded_candidate))

    os.remove(input_file)
    shutil.rmtree(output_folder)

    return solutions


def make_dataset(N):
    all_solutions = []
    for _ in tqdm(range(N)):
        problem = make_solvable_problem(15 * 4 // 2, 15 * 4, 4, 4)
        solution = solve_problem(problem)
        all_solutions.extend(solution)
    return all_solutions


def save_dataset(dataset, dir):
    torch.save(dataset, os.path.join(dir, f"{uuid4()}.pt"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python make_data.py N DIR")
        sys.exit(1)

    while True:
        num_solves = int(sys.argv[1])
        dir = sys.argv[2]
        save_dataset(make_dataset(num_solves), dir)
