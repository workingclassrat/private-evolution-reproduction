"""private evolution pipeline used by both run_unconditional.py and run_conditional.py"""

from config import NUM_SYNTHETIC_SAMPLES, NUM_ITERATIONS, NOISE_MULTIPLIER, HISTOGRAM_THRESHOLD, DELTA, VARIATION_SCHEDULE, TIMESTEP_RESPACING, DIFFUSION_BATCH_SIZE, EMBEDDING_BATCH_SIZE, LOOKAHEAD_DEGREE

import os
from pe.api.image import ImprovedDiffusion270M
from pe.embedding.image import Inception
from pe.histogram import NearestNeighbors
from pe.population import PEPopulation
from pe.logging import setup_logging
from pe.runner import PE
from pe.callback import SaveCheckpoints, SampleImages, ComputeFID
from pe.logger import ImageFile, CSVPrint, LogPrint
from pe.dp.gaussian import compute_epsilon

def run_pe_pipeline(data, exp_folder, num_synthetic_samples=None):
    num_synthetic_samples = num_synthetic_samples or NUM_SYNTHETIC_SAMPLES

    os.makedirs(exp_folder, exist_ok=True)
    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    epsilon = compute_epsilon(
        noise_multiplier=NOISE_MULTIPLIER,
        num_steps=NUM_ITERATIONS,
        delta=DELTA,
    )
    print(f"epsilon = {epsilon}")

    api = ImprovedDiffusion270M(
        variation_degrees=VARIATION_SCHEDULE[: NUM_ITERATIONS + 1],
        timestep_respacing=TIMESTEP_RESPACING,
        batch_size=DIFFUSION_BATCH_SIZE,
    )
    embedding = Inception(res=32, batch_size=EMBEDDING_BATCH_SIZE)
    histogram = NearestNeighbors(
        embedding=embedding,
        mode="L2",
        lookahead_degree=LOOKAHEAD_DEGREE,
        api=api,
        voting_details_log_folder=os.path.join(exp_folder, "voting_details"),
    )
    population = PEPopulation(api=api, histogram_threshold=HISTOGRAM_THRESHOLD)

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    sample_images = SampleImages(num_images_per_class=num_synthetic_samples)
    compute_fid = ComputeFID(priv_data=data, embedding=embedding)

    pe_runner = PE(
        priv_data=data,
        population=population,
        histogram=histogram,
        callbacks=[save_checkpoints, sample_images, compute_fid],
        loggers=[ImageFile(output_folder=exp_folder), CSVPrint(output_folder=exp_folder), LogPrint()],
    )

    try:
        pe_runner.run(
            num_samples_schedule=[num_synthetic_samples] * (NUM_ITERATIONS + 1),
            delta=DELTA,
            noise_multiplier=NOISE_MULTIPLIER,
            checkpoint_path=os.path.join(exp_folder, "checkpoint"),
        )
    except Exception as e:
        print(f"PE run failed: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()

    print(f"\nOutput folder: {exp_folder}")
    for root, dirs, files in os.walk(exp_folder):
        for f in files:
            print(os.path.join(root, f))

    return exp_folder
