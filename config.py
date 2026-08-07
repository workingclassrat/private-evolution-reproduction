import math

BASE_DIR = "/path/to/reproduction"
DATA_DIR = f"{BASE_DIR}/data"
RESULTS_DIR = f"{BASE_DIR}/results"

NUM_PRIVATE_IMAGES = 50000
NUM_SYNTHETIC_SAMPLES = 50000
NUM_ITERATIONS = 5
T_PARAM = 5
NOISE_MULTIPLIER = T_PARAM * math.sqrt(2)  #σ = t√2
HISTOGRAM_THRESHOLD = 2 * T_PARAM  #H = 2t
DELTA = 1e-5

VARIATION_SCHEDULE = [98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60]
TIMESTEP_RESPACING = 100
DIFFUSION_BATCH_SIZE = 2000
EMBEDDING_BATCH_SIZE = 2000
LOOKAHEAD_DEGREE = 8

CIFAR10_CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def build_exp_folder(mode):  #mode: conditional/unconditional
    return (
        f"{RESULTS_DIR}/{mode}_n{NUM_PRIVATE_IMAGES}"
        f"_iter{NUM_ITERATIONS}_syn{NUM_SYNTHETIC_SAMPLES}"
        f"_H{HISTOGRAM_THRESHOLD}_sigma{NOISE_MULTIPLIER:.2f}"
    )
