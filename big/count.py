import torch
import os
import sys

folder = sys.argv[1]

total = 0
for file in os.listdir(folder):
    dataset = torch.load(os.path.join(folder, file))
    total += len(dataset)

print(total)
