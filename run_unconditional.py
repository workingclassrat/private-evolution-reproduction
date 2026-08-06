"""unconditional version of private evolution"""

from config import DATA_DIR, RESULTS_DIR, NUM_PRIVATE_IMAGES, NUM_SYNTHETIC_SAMPLES, NUM_ITERATIONS, NOISE_MULTIPLIER, HISTOGRAM_THRESHOLD

import numpy as np
import pandas as pd
from pe.data import Data
from pe.constant.data import IMAGE_DATA_COLUMN_NAME, LABEL_ID_COLUMN_NAME

from pe_pipeline import run_pe_pipeline

CIFAR_NPZ_PATH = f"{DATA_DIR}/cifar10_unconditional_{NUM_PRIVATE_IMAGES}.npz"

if __name__ == "__main__":
    loaded = np.load(CIFAR_NPZ_PATH)
    images = loaded["images"][:NUM_PRIVATE_IMAGES]
    print(f"Loaded: {images.shape[0]} images, shape {images.shape[1:]}")

    df = pd.DataFrame({IMAGE_DATA_COLUMN_NAME: list(images)})
    df[LABEL_ID_COLUMN_NAME] = 0
    metadata = {"label_info": [{"name": "unconditional"}]}
    data = Data(data_frame=df, metadata=metadata)

    exp_folder = (
        f"{RESULTS_DIR}/unconditional_n{NUM_PRIVATE_IMAGES}"
        f"_iter{NUM_ITERATIONS}_syn{NUM_SYNTHETIC_SAMPLES}"
        f"_H{HISTOGRAM_THRESHOLD}_sigma{NOISE_MULTIPLIER:.2f}"
    )
    run_pe_pipeline(data, exp_folder)
