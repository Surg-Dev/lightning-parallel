import torch
from make_data import N, START, END, GROUND, WALL, EMPTY
from test import NET_EMPTY, NET_BOLT, NET_CANDIDATE, NET_WALL, NET_GROUND, draw_map
import numpy as np
import sys
from train_wall import build_unet
from evaluate_no_wall import load_ppm, update_candidates


def simulate(model, map, start, end, device, eta=2):
    candidates = set()
    update_candidates(candidates, map, start[0], start[1])

    map = torch.from_numpy(map)

    i = 0
    while candidates:
        x = torch.zeros((1, 2, N, N), device=device)
        # first channel is wall/empty, second channel is bolt/ground.
        x[:, 0, :, :] = map == NET_EMPTY
        x[:, 1, :, :] = map == NET_GROUND

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
