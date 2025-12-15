import argparse
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from PlantCalibrationModel import PlantCalibrationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_policy_action(policy_name: str, rng: np.random.Generator):
    """
    Returns an action vector based on the policy name.
    Action format: [red_coef, white_coef, blue_coef]
    """
    if policy_name == "Uniform Dirichlet":
        return rng.dirichlet([1, 1, 1])
    elif policy_name == "Discrete Random":
        options = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return rng.choice(options)
    elif policy_name == "Constant Red":
        return np.array([1.0, 0.0, 0.0])
    elif policy_name == "Constant White":
        return np.array([0.0, 1.0, 0.0])
    elif policy_name == "Constant Blue":
        return np.array([0.0, 0.0, 1.0])
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="plant-data/mixed-v19")
    parser.add_argument("--K", type=int, default=3, help="Number of neighbors")
    parser.add_argument(
        "--max_state_dist", type=float, default=0.5, help="Max state distance"
    )
    parser.add_argument(
        "--max_action_dist", type=float, default=0.1, help="Max action distance"
    )
    parser.add_argument("--steps", type=int, default=13, help="Rollout steps")
    parser.add_argument(
        "--num_rollouts", type=int, default=100, help="Number of rollouts"
    )
    parser.add_argument(
        "--output_plot", type=str, default="results/policy_comparison_calibration.png"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_plot).parent.mkdir(parents=True, exist_ok=True)

    # Initialize Model
    logger.info("Initializing PlantCalibrationModel...")
    env = PlantCalibrationModel(
        dataset_id=args.dataset_id,
        k=args.K,
        max_state_dist=args.max_state_dist,
        max_action_dist=args.max_action_dist,
    )

    policies = [
        "Uniform Dirichlet",
        "Discrete Random",
        "Constant Red",
        "Constant White",
        "Constant Blue",
    ]

    results = []
    returns_results = []


    for policy in policies:
        logger.info(f"Running rollouts for policy: {policy}")
        policy_rng = np.random.default_rng(args.seed)

        for i in tqdm(range(args.num_rollouts), desc=policy):
            obs, info = env.reset(seed=args.seed + i)

            current_area = info["area"]
            initial_area = current_area

            # Record initial area (Step 0)
            results.append(
                {
                    "Policy": policy,
                    "Step": 0,
                    "Area": current_area,
                    "RolloutID": i,
                }
            )

            current_return = 0.0

            for t in range(1, args.steps + 1):
                # Generate Action
                action_vec = get_policy_action(policy, policy_rng)

                # Step
                next_obs, reward, terminated, truncated, info = env.step(action_vec)

                current_area = info["area"]

                # Record result
                results.append(
                    {
                        "Policy": policy,
                        "Step": t,
                        "Area": current_area,
                        "RolloutID": i,
                    }
                )

                current_return += reward

                if terminated or truncated:
                    break

            # Metrics
            growth = current_area / initial_area if initial_area > 0 else 0.0

            returns_results.append(
                {
                    "Policy": policy,
                    "Value": current_return,
                    "Metric": "Dataset Return",
                    "RolloutID": i,
                }
            )

            returns_results.append(
                {
                    "Policy": policy,
                    "Value": growth,
                    "Metric": "Growth Ratio",
                    "RolloutID": i,
                }
            )

    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    df_returns = pd.DataFrame(returns_results)

    logger.info("Plotting results...")

    # Define custom color palette
    custom_palette = {
        "Uniform Dirichlet": "tab:orange",
        "Discrete Random": "tab:green",
        "Constant Red": "red",
        "Constant White": "black",
        "Constant Blue": "blue",
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # Seaborn lineplot for Areas
    sns.lineplot(
        data=df_results,
        x="Step",
        y="Area",
        hue="Policy",
        palette=custom_palette,
        estimator="mean",
        errorbar=("ci", 95),
        ax=axes[0],
    )

    axes[0].set_title("Policy Comparison (Calibration Model) - Area Trajectories")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Plant Area")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="Policy")

    # Seaborn barplot for Dataset Returns
    sns.barplot(
        data=df_returns[df_returns["Metric"] == "Dataset Return"],
        x="Policy",
        y="Value",
        hue="Policy",
        palette=custom_palette,
        errorbar=("ci", 95),
        ax=axes[1],
    )

    axes[1].set_title("Policy Comparison - Total Dataset Return (Reward Sum)")
    axes[1].set_xlabel("Policy")
    axes[1].set_ylabel("Return")
    axes[1].grid(True, alpha=0.3)

    # Seaborn barplot for Growth Ratio
    sns.barplot(
        data=df_returns[df_returns["Metric"] == "Growth Ratio"],
        x="Policy",
        y="Value",
        hue="Policy",
        palette=custom_palette,
        errorbar=("ci", 95),
        ax=axes[2],
    )

    axes[2].set_title("Policy Comparison - Growth Ratio (Final / Initial)")
    axes[2].set_xlabel("Policy")
    axes[2].set_ylabel("Growth Ratio")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output_plot)
    logger.info(f"Saved plot to {args.output_plot}")


if __name__ == "__main__":
    main()
