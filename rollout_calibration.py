import argparse
import glob
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from etils.etqdm.tqdm_utils import tqdm

from calibration_model import CalibrationModel
from plot import plot_trajectories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Path resolution cache
_PATH_CACHE = {}


def get_image_root(experiment, zone):
    """
    Finds the root directory for images given experiment and zone.
    Returns the path up to 'processed/v17', so specific images can be appended.
    """
    key = (experiment, zone)
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]

    # Generic pattern to handle different Pods and Treatment names
    # /data/plant-rl/online/E{exp}/P*/*/*zone{zone:02d}/processed/v17

    # Try generic online path
    # Zone needs zero padding
    zone_str = f"{int(zone):02d}"

    # Glob pattern
    # We are looking for something like:
    # /data/plant-rl/online/E11/P1/DiscreteRandom1/alliance-zone01/processed/v17

    # Use wildcards for Pod (P*), Treatment (*), and Zone Prefix (*zone)
    base_pattern = (
        f"/data/plant-rl/online/E{experiment}/P*/*/*zone{zone_str}/processed/v17"
    )
    matches = glob.glob(base_pattern)

    if matches:
        # Sort to be deterministic, pick first
        path = matches[0]
        _PATH_CACHE[key] = path
        return path

    logger.warning(
        f"Could not find image root for E{experiment} Z{zone} with pattern {base_pattern}"
    )
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="/data/plant-rl/offline/v18/mixed-v18.parquet"
    )
    parser.add_argument(
        "--norm_path",
        type=str,
        default="/data/plant-rl/offline/v18/normalization-stats-v18.json",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/plant-rl/offline",
        help="Fallback root for images if dynamic resolution fails",
    )
    parser.add_argument("--K", type=int, default=3, help="Number of neighbors")
    parser.add_argument(
        "--state_threshold", type=float, default=0.8, help="State similarity threshold"
    )
    parser.add_argument(
        "--action_threshold",
        type=float,
        default=0.6,
        help="Action similarity threshold",
    )
    parser.add_argument("--steps", type=int, default=13, help="Rollout steps")
    parser.add_argument(
        "--num_rollouts", type=int, default=5, help="Number of rollouts"
    )
    parser.add_argument("--output_plot", type=str, default="rollout_results.png")
    parser.add_argument("--output_image_plot", type=str, default="rollout_images.png")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Configuration
    state_cols = ["cls_token", "clean_area", "wall_time"]
    action_cols = ["action_coefficients"]

    # Initialize Model
    logger.info("Initializing Calibration Model...")
    model = CalibrationModel(
        data_path=args.data_path,
        normalization_path=args.norm_path,
        state_cols=state_cols,
        action_cols=action_cols,
        K=args.K,
        state_threshold=args.state_threshold,
        action_threshold=args.action_threshold,
        seed=args.seed,
    )

    # Pick random start states from the dataset
    logger.info("Starting rollouts...")

    plt.figure(figsize=(10, 6))

    all_trajectories = []

    for i in tqdm(range(args.num_rollouts)):
        # Sample initial state
        state, seen = model.sample_initial_state(seed=i)

        # Storage for plotting
        areas = [state["clean_area"]]

        traj_data = []  # List of dicts

        rewards = []
        action = {"action_coefficients": [1.0, 0.0, 0.0]}

        # We need the image/info for the *initial* state, but sample_initial_state doesn't return it
        # For now, we will just plot the sequence starting from step 0 (which is the next state from initial)
        # OR we could modify sample to return it.
        # Given constraints, let's just plot the trajectory steps (next_states).

        for t in range(args.steps):
            next_state, reward, term, info, seen = model.predict(state, action, seen)

            image_path = None
            if info and info["image_path"]:
                # specific relative path
                rel_path = info["image_path"]
                exp = info.get("experiment")
                zone = info.get("zone")

                if exp is not None and zone is not None:
                    root = get_image_root(exp, zone)
                    if root:
                        image_path = f"{root}/{rel_path}"
                    else:
                        # Fallback
                        image_path = f"{args.data_root}/{rel_path}"
                else:
                    image_path = f"{args.data_root}/{rel_path}"

            step_data = {
                "dataset_area": next_state["clean_area"]
                if next_state
                else 0,  # Renamed to match plot expectation
                "image_path": image_path,
            }
            traj_data.append(step_data)

            if next_state is None:
                logger.info(f"Rollout {i} terminated early at step {t} (threshold)")
                break

            if term:
                logger.info(f"Rollout {i} reached terminal state at step {t}")
                areas.append(next_state["clean_area"])
                break

            state = next_state
            areas.append(state["clean_area"])
            rewards.append(reward)

        all_trajectories.append(traj_data)
        plt.plot(areas, label=f"Rollout {i}")

    plt.title("Calibration Model Rollouts (Area)")
    plt.xlabel("Step")
    plt.ylabel("Area")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.output_plot)
    logger.info(f"Saved rollout plot to {args.output_plot}")

    # Plot images
    plot_trajectories(
        all_trajectories,
        data_dir="",  # Paths are absolute
        output_dir=Path("."),  # Save to current dir
        filename=args.output_image_plot,
    )


if __name__ == "__main__":
    main()
