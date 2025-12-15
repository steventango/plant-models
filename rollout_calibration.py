import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from plant_models import PlantCalibrationModel
from plot import plot_trajectories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstantAgent:
    def __init__(self, action):
        self.action = action

    def act(self, _):
        return self.action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="plant-data/mixed-v19")
    parser.add_argument("--K", type=int, default=3, help="Number of neighbors")
    parser.add_argument(
        "--max_stat_dist", type=float, default=3.0, help="Max stat distance"
    )
    parser.add_argument(
        "--max_emb_dist", type=float, default=1.0, help="Max embedding distance"
    )
    max_action_dist = np.linalg.norm(np.array([0, 1, 0] - np.ones(3) / 3))
    parser.add_argument(
        "--max_action_dist", type=float, default=max_action_dist, help="Max action distance"
    )
    parser.add_argument("--steps", type=int, default=13, help="Rollout steps")
    parser.add_argument(
        "--num_rollouts", type=int, default=10, help="Number of rollouts"
    )
    parser.add_argument(
        "--output_plot", type=str, default="results/rollout_results.png"
    )
    parser.add_argument(
        "--output_image_plot", type=str, default="results/rollout_images.png"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    agent = ConstantAgent(action=np.array([0.0, 0.0, 1.0], dtype=np.float32))

    # Initialize Model
    logger.info("Initializing PlantCalibrationModel...")
    env = PlantCalibrationModel(
        dataset_id=args.dataset_id,
        k=args.K,
        max_stat_dist=args.max_stat_dist,
        max_emb_dist=args.max_emb_dist,
        max_action_dist=args.max_action_dist,
    )

    # Pick random start states from the dataset
    logger.info("Starting rollouts...")

    plt.figure(figsize=(10, 6))

    all_trajectories = []

    for i in tqdm(range(args.num_rollouts)):
        obs, info = env.reset(seed=i + args.seed)

        clean_area_idx = 1
        areas = [info["area"]]
        rewards = []
        traj_data = []

        for t in range(args.steps):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            step_data = {
                "dataset_area": next_obs[clean_area_idx],
                "image_path": info.get("image_path"),
            }
            traj_data.append(step_data)

            obs = next_obs
            areas.append(info["area"])
            rewards.append(reward)
            
            if terminated:
                logger.info(f"Rollout {i} terminated at step {t}")
                if "error" in info:
                    logger.info(f"Termination reason: {info['error']}")
                break

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
        output_dir=Path("."),
        filename=args.output_image_plot,
    )


if __name__ == "__main__":
    main()
