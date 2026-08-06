"""unconditional generation: no classes"""

from config import DATA_DIR, NUM_PRIVATE_IMAGES

import os
import numpy as np
from datasets import load_dataset

SAVE_PATH = f"{DATA_DIR}/cifar10_unconditional_{NUM_PRIVATE_IMAGES}.npz"

if __name__ == "__main__":
    ds = load_dataset("uoft-cs/cifar10", split="train", streaming=True)
    first_n = list(ds.take(NUM_PRIVATE_IMAGES))

    images = np.stack([np.array(ex["img"]) for ex in first_n])
    labels = np.array([ex["label"] for ex in first_n])

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez(SAVE_PATH, images=images, labels=labels)
