import torch
import time
from train import build_unet
from make_data import N

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

print(sum(nums) / len(nums))