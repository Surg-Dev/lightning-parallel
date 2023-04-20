import torch
from make_data import N, START, END, GROUND, WALL, EMPTY
from test import NET_EMPTY, NET_BOLT, NET_CANDIDATE, NET_WALL, NET_GROUND, draw_map
import numpy as np
import sys
from train_no_wall import build_unet
import poisson_disc as pd


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


def simulate(model, map, start, end, device, eta=1):
    candidates = set()
    update_candidates(candidates, map, start[0], start[1])

    points = pd.Bridson_sampling(radius=0.1)
    for point in points:
        y, x = int(point[0] * N), int(point[1] * N)
        if map[y, x] == NET_EMPTY:
            map[y, x] = NET_GROUND

    i = 0
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
        candidates.remove(chosen)
        update_candidates(candidates, map, chosen[0], chosen[1])

        if i % 500 == 0:
            import matplotlib.pyplot as plt

            plt.imshow(draw_map(map))
            plt.show()

        if chosen == end:
            print("Found solution")
            return

        i += 1

    print("No solution found")


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

    simulate(model, map, start, end, device)
