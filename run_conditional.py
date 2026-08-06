"""conditional version of private evolution: one histogram per class"""

from config import DATA_DIR, RESULTS_DIR, NUM_PRIVATE_IMAGES, NUM_SYNTHETIC_SAMPLES, NUM_ITERATIONS, NOISE_MULTIPLIER, HISTOGRAM_THRESHOLD

import numpy as np
import pandas as pd
from pe.data import Data
from pe.constant.data import IMAGE_DATA_COLUMN_NAME, LABEL_ID_COLUMN_NAME

from pe_pipeline import run_pe_pipeline

PER_CLASS = NUM_PRIVATE_IMAGES // 10
CIFAR_NPZ_PATH = f"{DATA_DIR}/cifar10_conditional_{PER_CLASS}pc.npz"

if __name__ == "__main__":
    loaded = np.load(CIFAR_NPZ_PATH)
    images, labels = loaded["images"], loaded["labels"]
    images, labels = images[:NUM_PRIVATE_IMAGES], labels[:NUM_PRIVATE_IMAGES]
    print(f"Loaded: {images.shape[0]} images, shape {images.shape[1:]}")

    present = sorted(set(labels.tolist()))
    if len(present) < 10:
        missing = [CIFAR10_CLASS_NAMES[m] for m in sorted(set(range(10)) - set(present))]
        raise ValueError(f"Missing classes: {missing}. Increase NUM_PRIVATE_IMAGES and re-download.")

    df = pd.DataFrame({IMAGE_DATA_COLUMN_NAME: list(images), LABEL_ID_COLUMN_NAME: labels})
    metadata = {"label_info": [{"name": n} for n in CIFAR10_CLASS_NAMES]}
    data = Data(data_frame=df, metadata=metadata)

    exp_folder = (
        f"{RESULTS_DIR}/conditional_n{NUM_PRIVATE_IMAGES}"
        f"_iter{NUM_ITERATIONS}_syn{NUM_SYNTHETIC_SAMPLES}"
        f"_H{HISTOGRAM_THRESHOLD}_sigma{NOISE_MULTIPLIER:.2f}"
    )
    run_pe_pipeline(data, exp_folder)
