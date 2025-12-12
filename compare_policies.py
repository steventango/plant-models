import argparse
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from calibration_model import CalibrationModel

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
    parser.add_argument(
        "--data_path", type=str, default="/data/plant-rl/offline/v18/mixed-v18.parquet"
    )
    parser.add_argument(
        "--norm_path",
        type=str,
        default="/data/plant-rl/offline/v18/normalization-stats-v18.json",
    )
    parser.add_argument("--K", type=int, default=3, help="Number of neighbors")
    parser.add_argument(
        "--state_threshold", type=float, default=0.9, help="State similarity threshold"
    )
    parser.add_argument(
        "--action_threshold",
        type=float,
        default=0.6,
        help="Action similarity threshold",
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

    policies = [
        "Uniform Dirichlet",
        "Discrete Random",
        "Constant Red",
        "Constant White",
        "Constant Blue",
    ]

    # Pre-sample initial states so all policies start from the same conditions
    logger.info(f"Sampling {args.num_rollouts} initial states...")
    initial_states = []
    for i in range(args.num_rollouts):
        state, seen = model.sample_initial_state(seed=args.seed + i)
        initial_states.append(
            (state, seen)
        )  # Store independent copies/seeds if needed?
        # sample_initial_state returns a new 'seen' set with just that index.
        # But we need fresh 'seen' for each rollout.
        # So we just store the initial state dictionary. 'seen' needs to be reset per rollout.

    results = []
    returns_results = []

    rng = np.random.default_rng(args.seed)

    for policy in policies:
        logger.info(f"Running rollouts for policy: {policy}")

        for i in tqdm(range(args.num_rollouts)):
            init_state_dict, init_seen = initial_states[i]
            state = init_state_dict.copy()
            seen = init_seen.copy()

            # Record initial area (Step 0)
            results.append(
                {
                    "Policy": policy,
                    "Step": 0,
                    "Area": state["clean_area"],
                    "RolloutID": i,
                }
            )

            current_return = 0.0

            for t in range(1, args.steps + 1):
                # Generate Action
                action_vec = get_policy_action(policy, rng)
                action = {"action_coefficients": action_vec}

                # Predict Next State
                next_state, reward, term, info, seen = model.predict(
                    state, action, seen
                )

                if reward is None:
                    current_return = model.default_return
                else:
                    current_return += reward

                if next_state is None:
                    break

                # Record result
                results.append(
                    {
                        "Policy": policy,
                        "Step": t,
                        "Area": next_state["clean_area"],
                        "RolloutID": i,
                    }
                )

                if term:
                    break

                state = next_state

            growth = (
                state["clean_area"] / init_state_dict["clean_area"]
                if init_state_dict["clean_area"] > 0
                else 0.0
            )

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
