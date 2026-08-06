"""conditional generation: even number of CIFAR-10 images from each of the 10 classes"""

from config import DATA_DIR, NUM_PRIVATE_IMAGES

import os
import numpy as np
from datasets import load_dataset
from collections import defaultdict

PER_CLASS = NUM_PRIVATE_IMAGES // 10
SAVE_PATH = f"{DATA_DIR}/cifar10_conditional_{PER_CLASS}pc.npz"

if __name__ == "__main__":
    ds = load_dataset("uoft-cs/cifar10", split="train", streaming=True)

    collected = defaultdict(list)
    for ex in ds:
        label = ex["label"]
        if len(collected[label]) < PER_CLASS:
            collected[label].append(np.array(ex["img"]))
        if all(len(collected[c]) >= PER_CLASS for c in range(10)):
            break
          
    images, labels = [], []
    for c in range(10):
        images.extend(collected[c])
        labels.extend([c] * PER_CLASS)

    images = np.stack(images)
    labels = np.array(labels)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez(SAVE_PATH, images=images, labels=labels)
