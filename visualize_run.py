"""
generate a private image grid, synthetic image grid for each iteration, and an FID vs privacy cost (epsilon) graph
usage: python visualize_run.py --exp_folder <path> --npz_path <path>
"""

from config import NOISE_MULTIPLIER, DELTA

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pe.dp.gaussian import compute_epsilon

parser = argparse.ArgumentParser()
parser.add_argument("--exp_folder", required=True, help="Path to the run's results folder")
parser.add_argument("--npz_path", required=True, help="Path to the .npz this run's private data came from")
args = parser.parse_args()

viz_out = os.path.join(args.exp_folder, "viz")
os.makedirs(viz_out, exist_ok=True)

#detect real number of iterations from checkpoint folders
checkpoint_dir = os.path.join(args.exp_folder, "checkpoint")
num_iterations = len([d for d in os.listdir(checkpoint_dir)
                       if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.isdigit()]) - 1

"""private image grid"""
loaded = np.load(args.npz_path)
real_images = loaded["images"]
n = len(real_images)
ncols = math.ceil(math.sqrt(n))
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
axes = axes.flatten()
for i, ax in enumerate(axes):
    if i < n:
        ax.imshow(real_images[i])
    ax.axis("off")
fig.suptitle(f"CIFAR-10 images (n={n})")
plt.tight_layout()
fig.savefig(os.path.join(viz_out, "private_images.png"))
plt.close(fig)
print(f"Saved {viz_out}/private_images.png")

"""synthetic image grid (one image per iteration)"""
for i in range(num_iterations + 1):
    img_path = os.path.join(args.exp_folder, "image_sample", f"{i:09d}.png")
    if not os.path.exists(img_path):
        print(f"Iteration {i}: image not found, skipping")
        continue
    label = "Initial population (RANDOM_API only)" if i == 0 else f"After {i} PE iteration(s)"
    img = Image.open(img_path)
    fig = plt.figure(figsize=(4, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(label)
    out_path = os.path.join(viz_out, f"synthetic_iter{i}.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")

"""FID vs privacy cost (epsilon) graph"""
fid_csv_path = os.path.join(args.exp_folder, "fid_PE.EMBEDDING.Inception.csv") #reuses FID values computed during run
if os.path.exists(fid_csv_path):
    fid_csv = pd.read_csv(fid_csv_path, header=None, names=["iteration", "fid"])
    fid_csv = fid_csv[fid_csv["iteration"] >= 1].sort_values("iteration")
    epsilons = [compute_epsilon(noise_multiplier=NOISE_MULTIPLIER, num_steps=int(i), delta=DELTA)
                for i in fid_csv["iteration"]]
    plt.figure()
    plt.plot(epsilons, fid_csv["fid"].tolist(), marker="o")
    plt.xlabel("epsilon")
    plt.ylabel("FID")
    plt.title("Privacy-quality trade-off")
    plt.savefig(os.path.join(viz_out, "epsilon_vs_fid.png"))
    print(f"Saved {viz_out}/epsilon_vs_fid.png")
else:
    print(f"No FID CSV found at {fid_csv_path}; skipping epsilon vs. FID graph")
