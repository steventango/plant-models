import argparse
import logging

import matplotlib.pyplot as plt
from etils.etqdm.tqdm_utils import tqdm
import numpy as np

from PlantCalibrationModel import PlantCalibrationModel
# from plot import plot_trajectories

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
        "--max_state_dist", type=float, default=0.3, help="Max state distance"
    )
    parser.add_argument(
        "--max_action_dist", type=float, default=0.1, help="Max action distance"
    )
    parser.add_argument("--steps", type=int, default=13, help="Rollout steps")
    parser.add_argument(
        "--num_rollouts", type=int, default=64, help="Number of rollouts"
    )
    parser.add_argument("--output_plot", type=str, default="rollout_results.png")
    parser.add_argument("--output_image_plot", type=str, default="rollout_images.png")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    agent = ConstantAgent(action=np.array([0.0, 1.0, 0.0], dtype=np.float32))

    # Initialize Model
    logger.info("Initializing PlantCalibrationModel...")
    env = PlantCalibrationModel(
        dataset_id=args.dataset_id,
        k=args.K,
        max_state_dist=args.max_state_dist,
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
                "image_path": None,
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
    # plot_trajectories(
    #     all_trajectories,
    #     data_dir="",  # Paths are absolute
    #     output_dir=Path("."),  # Save to current dir
    #     filename=args.output_image_plot,
    # )


if __name__ == "__main__":
    main()
