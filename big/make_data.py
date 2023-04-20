import numpy as np
import random

import os
import sys
import tempfile

from tqdm import tqdm
from uuid import uuid4
import torch

N = 256

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
        x, y = random.randint(0, N - w), random.randint(0, N - h)
        if not has_problematic_overlap(problem, (x, y, w, h), type):
            problem[y : y + h, x : x + w] = type
            break


def make_problem(max_ground, max_wall, max_ground_len, max_wall_len):
    problem = np.full((N, N), EMPTY, dtype=np.uint32)
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
    for i in range(N):
        for j in range(N):
            if problem[i, j] != WALL:
                G.add_node((i, j))

    for i in range(N):
        for j in range(N):
            if problem[i, j] != WALL:
                if i + 1 < N and problem[i + 1, j] != WALL:
                    G.add_edge((i, j), (i + 1, j))
                if j + 1 < N and problem[i, j + 1] != WALL:
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
        f.write(f"{N} {N}\n".encode("ascii"))
        f.write(b"255\n")
        for row in problem:
            for pixel in row:
                f.write(bytes((pixel >> 16 & 0xFF, pixel >> 8 & 0xFF, pixel & 0xFF)))


def solve_problem(filename):
    temp_folder = tempfile.mkdtemp()

    # os.system(f"../lumosquad-0.1/lumosquad {filename} {temp_folder}/out 1")
    # instead, use
    import subprocess

    subprocess.run(
        ["../lumosquad-0.1/lumosquad", filename, f"{temp_folder}/out", "1"],
        stdout=subprocess.PIPE,
    )

    files = os.listdir(temp_folder)
    files.sort()

    pairs = zip(files[::2], files[1::2])

    def load_bolt(bolt_file):
        result = np.zeros((N, N), dtype=np.uint8)
        with open(bolt_file, "rb") as f:
            for i in range(N):
                for j in range(N):
                    result[i, j] = int.from_bytes(f.read(4), "little") > 0
        return result

    def load_candidate(candidate_file):
        result = np.zeros((N, N), dtype=np.float32)
        with open(candidate_file, "rb") as f:
            for i in range(N):
                for j in range(N):
                    result[i, j] = int.from_bytes(f.read(4), "little")
        return result

    solutions = []
    for bolt, candidate in pairs:
        solutions.append(
            (
                load_bolt(os.path.join(temp_folder, bolt)),
                load_candidate(os.path.join(temp_folder, candidate)),
            )
        )

    return solutions


def make_dataset(N):
    all_solutions = []
    for _ in tqdm(range(N)):
        problem = make_solvable_problem(30, 30, 30, 30)
        temp = tempfile.mktemp()
        save_problem(problem, temp)
        solution = solve_problem(temp)
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
