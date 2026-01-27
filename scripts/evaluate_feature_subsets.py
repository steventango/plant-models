import argparse
import logging
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pathlib import Path
from plant_models import PlantCalibrationModel, PlantDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Feature Indices ---
SOLIDITY_IDX = 3
LOG_AREA_IDX = 28
PCA_START_IDX = 38
OPTIMAL_PCA_COUNT = 4

WALL_TIME_IDX = 0
DAYS_STERILIZATION_IDX = 29
DAYS_TRANSPLANT_IDX = 30
DAYS_DOME_REMOVAL_IDX = 31
DAYS_WATERING_IDX = 32
LITERS_PER_POT_IDX = 33
RED_TRACE_IDX = 34
WHITE_TRACE_IDX = 35
BLUE_TRACE_IDX = 36


def estimate_subset_sigma(dataset, stat_indices, emb_weight, n_samples=1000, seed=42):
    """Estimate median distance (sigma) for a specific feature subset using TRAIN data."""
    rng = np.random.RandomState(seed)
    n_data = dataset.X_stat_norm.shape[0]
    idx1 = rng.choice(n_data, min(n_samples, n_data))
    idx2 = rng.choice(n_data, min(n_samples, n_data))

    total_dist_sq = np.zeros(len(idx1))

    if len(stat_indices) > 0:
        s1 = np.array(dataset.X_stat_norm)[idx1][:, stat_indices]
        s2 = np.array(dataset.X_stat_norm)[idx2][:, stat_indices]
        total_dist_sq += np.sum((s1 - s2) ** 2, axis=1)

    if emb_weight > 0:
        e1 = np.array(dataset.X_emb_norm)[idx1]
        e2 = np.array(dataset.X_emb_norm)[idx2]
        total_dist_sq += (1.0 - np.sum(e1 * e2, axis=1)) ** 2

    sigma = np.median(np.sqrt(total_dist_sq))
    return float(sigma) if sigma > 1e-6 else 1.0


def run_evaluation(env, val_dataset, max_actions: int = 13):
    """Evaluate reconstruction on VALIDATION episodes using the env (initialized with TRAIN data)."""
    results = []
    episodes = list(val_dataset.dataset)

    for ep in episodes:
        action_seq = ep.actions[:max_actions]
        gt_return = np.sum(ep.rewards[:max_actions])

        env.reset()
        env.current_state = np.array(ep.observations[0])
        env.seen_mask = jnp.zeros(env.data.X_stat.shape[0], dtype=bool)
        env.current_return = 0.0

        steps = 0
        for action in action_seq:
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated:
                break

        results.append(
            {
                "steps_pct": steps / len(action_seq),
                "return_error": abs(env.current_return - gt_return),
                "failed": "error" in info,
            }
        )

    df = pd.DataFrame(results)
    return {
        "mean_steps_pct": df["steps_pct"].mean(),
        "mean_return_error": df["return_error"].mean(),
        "failure_rate": df["failed"].mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=str, default="results/all_final_variations_sweep.csv"
    )
    args = parser.parse_args()

    # Base Features
    # PCA_4: Indices 38, 39, 40, 41
    pca_indices = list(range(PCA_START_IDX, PCA_START_IDX + 4))
    # LogArea: 28, Solidity: 3
    base_indices = pca_indices + [LOG_AREA_IDX, SOLIDITY_IDX]

    subsets = [
        # 1. Base
        {
            "name": "Base (PCA4+LogArea+Solidity)",
            "indices": base_indices,
            "emb_weight": 0.0,
        },
        # Base - PCA
        {
            "name": "Base - PCA",
            "indices": [LOG_AREA_IDX, SOLIDITY_IDX],
            "emb_weight": 0.0,
        },
        # 2. Base + liters_per_pot
        {
            "name": "Base + liters_per_pot",
            "indices": base_indices + [LITERS_PER_POT_IDX],
            "emb_weight": 0.0,
        },
        # # 3. Base + light traces (red, white, blue coef traces)
        # {
        #     "name": "Base + light_traces",
        #     "indices": base_indices + [RED_TRACE_IDX, WHITE_TRACE_IDX, BLUE_TRACE_IDX],
        #     "emb_weight": 0.0,
        # },
    ]

    threshold_factors = [1.0, 2.0, 3.0, 4.0, 5.0]

    Path("results").mkdir(parents=True, exist_ok=True)
    all_results = []

    for fold in range(5):
        logger.info(f"Evaluating Fold {fold}...")

        train_data_id = f"plant-data/mixed-fold{fold}-train-v23"
        val_data_id = f"plant-data/mixed-fold{fold}-val-v23"

        train_data = PlantDataset.get(train_data_id)
        val_data = PlantDataset.get(val_data_id)

        for subset in subsets:
            subset_sigma = estimate_subset_sigma(
                train_data, subset["indices"], subset["emb_weight"]
            )
            logger.info(f"  {subset['name']}, Sigma (Train): {subset_sigma:.4f}")

            for factor in threshold_factors:
                actual_threshold = factor * subset_sigma
                weights = np.zeros(48)
                for idx in subset["indices"]:
                    if idx < 48:
                        weights[idx] = 1.0

                env = PlantCalibrationModel(
                    dataset_id=train_data,
                    stat_weights=jnp.array(weights),
                    emb_weight=subset["emb_weight"],
                    max_stat_dist=actual_threshold,
                    max_emb_dist=10.0,
                    k=10,
                )

                metrics = run_evaluation(env, val_data)
                metrics.update(
                    {
                        "fold": fold,
                        "subset": subset["name"],
                        "factor": factor,
                        "actual_threshold": actual_threshold,
                        "sigma": subset_sigma,
                    }
                )
                all_results.append(metrics)
                logger.info(
                    f"    Factor: {factor}, Steps%: {metrics['mean_steps_pct']:.2%}, Error: {metrics['mean_return_error']:.4f}"
                )

    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)

    summary = (
        df.groupby(["subset", "factor"])
        .agg(
            {
                "mean_steps_pct": "mean",
                "mean_return_error": "mean",
                "failure_rate": "mean",
            }
        )
        .reset_index()
    )

    print("\nVariations Sweep Summary:")
    print(summary)

    best_configs = (
        summary.sort_values(
            ["mean_steps_pct", "mean_return_error"], ascending=[False, True]
        )
        .groupby("subset")
        .head(1)
    )
    best_configs = best_configs.sort_values("mean_return_error")
    print("\nBest per Configuration (Sorted by Error):")
    print(best_configs)


if __name__ == "__main__":
    main()
