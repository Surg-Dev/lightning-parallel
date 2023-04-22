import torch
import time
from train import build_unet


def benchmark(N):
    device = "cuda"

    model = build_unet()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    model.to(device)

    nums = []
    for _ in range(10000):
        rand = torch.rand((1, 1, N, N), device=device)
        start = time.time()
        model(rand)
        end = time.time()
        nums.append(end - start)

    print(f"Average time: {sum(nums) / len(nums)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 benchmark.py <N>")
        sys.exit(1)

    benchmark(int(sys.argv[1]))
