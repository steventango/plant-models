from etils.etqdm.tqdm_utils import tqdm
import argparse
import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import polars as pl
from flax import nnx
from sklearn.metrics.pairwise import cosine_similarity

from dataset import load_data, prepare_data
from network import MLP
from plot import plot_trajectories


def load_model(checkpoint_path: Path, input_dim: int, output_heads: dict[str, int]):
    """Load a model from checkpoint."""
    # Create model with dummy initialization
    temp_model = MLP(
        input_dim=input_dim,
        output_heads=output_heads,
        rngs=nnx.Rngs(0),
    )

    # Split to get graphdef and state
    graphdef, _ = nnx.split(temp_model)

    # Load checkpoint state
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(checkpoint_path.resolve())

    # Merge graphdef with loaded state
    model = nnx.merge(graphdef, ckpt["model"])
    return model


def policy_fn(policy_type: str, timestep: int = 0):
    """
    Generate action based on policy type.

    Args:
        policy_type: Type of policy ('white', 'red', 'blue', 'random', etc.)
        timestep: Current timestep (for time-dependent policies)

    Returns:
        Action as [red_coef, white_coef, blue_coef]
    """
    if policy_type == "white":
        return jnp.array([0.0, 1.0, 0.0])
    elif policy_type == "red":
        return jnp.array([1.0, 0.0, 0.0])
    elif policy_type == "blue":
        return jnp.array([0.0, 0.0, 1.0])
    elif policy_type == "random":
        # Random action
        action = np.random.dirichlet([1, 1, 1])
        return jnp.array(action)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def autoregressive_rollout(
    transition_model: MLP,
    decoder_model: MLP,
    initial_embedding: jnp.ndarray,
    policy_type: str,
    df: pl.DataFrame,
    rollout_strategy: str,
    K: int,
    rngs: nnx.Rngs,
):
    """
    Perform autoregressive rollout.

    Args:
        transition_model: Model that predicts next_embedding from (embedding, action)
        decoder_model: Model that predicts area from embedding
        initial_embedding: Starting embedding
        df: Dataset for k-nearest lookup
        rollout_strategy: Strategy for next state selection ('standard', 'k_nearest')
        policy_type: Type of policy to use
        K: Number of steps to rollout
        rngs: Random number generators

    Returns:
        List of (embedding, action, predicted_area) tuples for each timestep
    """
    trajectory = []
    current_embedding = initial_embedding

    # Pre-compute all embeddings if using k-nearest
    all_embeddings = None
    if rollout_strategy == "k_nearest":
        all_embeddings = np.stack(df["embedding"].to_list())

    for t in range(K):
        # Get action from policy
        action = policy_fn(policy_type, t)

        # Predict area from current embedding
        area_pred = decoder_model(current_embedding[None, :], rngs)["area"].squeeze()

        # Store current state
        trajectory.append(
            {
                "embedding": current_embedding,
                "action": action,
                "area": float(area_pred),
            }
        )

        # Predict next embedding
        model_input = jnp.concatenate([current_embedding, action])
        next_embedding_pred = transition_model(model_input[None, :], rngs)[
            "next_embedding"
        ].squeeze()

        if rollout_strategy == "standard":
            current_embedding = next_embedding_pred
        elif rollout_strategy == "k_nearest":
            # Find k=3 nearest embeddings
            similarities = cosine_similarity(
                next_embedding_pred.reshape(1, -1), all_embeddings
            )[0]

            # Get top k indices
            k = 3
            top_k_indices = np.argsort(similarities)[-k:]
            top_k_similarities = similarities[top_k_indices]

            # Calculate probabilities proportional to similarity
            # Ensure similarities are positive with softmax for probability calculation
            # (Cosine similarity is [-1, 1], but we expect positive for similar items)
            probs = np.exp(top_k_similarities) / np.sum(np.exp(top_k_similarities))

            # Sample one index
            chosen_idx = np.random.choice(top_k_indices, p=probs)
            current_embedding = jnp.array(all_embeddings[chosen_idx])
        else:
            raise ValueError(f"Unknown rollout strategy: {rollout_strategy}")

    return trajectory


def find_nearest_image(embedding: jnp.ndarray, df: pl.DataFrame):
    """
    Find the image in the dataset with the nearest cosine similarity to the embedding.

    Args:
        embedding: Target embedding
        df: DataFrame with embeddings and image paths

    Returns:
        Path to the nearest image
    """
    # Get all embeddings from dataset
    all_embeddings = np.stack(df["embedding"].to_list())

    # Compute cosine similarity
    similarities = cosine_similarity(embedding.reshape(1, -1), all_embeddings)[0]

    # Find index of maximum similarity
    nearest_idx = int(np.argmax(similarities))

    area = df["area"][nearest_idx]

    return df["image_path"][nearest_idx], similarities[nearest_idx], area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transition_checkpoint",
        type=str,
        default="results/transition/model",
        help="Path to transition model checkpoint",
    )
    parser.add_argument(
        "--decoder_checkpoint",
        type=str,
        default="results/decoder_area/model",
        help="Path to decoder model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/plant-rl/offline/cleaned_offline_dataset_daily_continuous_v15.parquet",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/rollout",
        help="Output directory for results",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=14,
        help="Number of rollout steps",
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=5,
        help="Number of rollout trajectories to generate",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="white",
        choices=["white", "red", "blue", "random"],
        help="Policy type",
    )
    parser.add_argument(
        "--rollout_strategy",
        type=str,
        default="standard",
        choices=["standard", "k_nearest"],
        help="Strategy for next state selection",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Set random seed
    np.random.seed(args.seed)

    # Load data
    logging.info(f"Loading data from {args.data_path}")
    df = load_data(args.data_path)
    df = prepare_data(df, mode="decoder")  # Use decoder mode to get clean data

    # Load models
    logging.info(f"Loading transition model from {args.transition_checkpoint}")
    transition_model = load_model(
        Path(args.transition_checkpoint),
        input_dim=768 + 3,  # embedding (768) + action (3)
        output_heads={"next_embedding": 768},
    )
    transition_model.eval()

    logging.info(f"Loading decoder model from {args.decoder_checkpoint}")
    decoder_model = load_model(
        Path(args.decoder_checkpoint),
        input_dim=768,  # embedding
        output_heads={"area": 1},
    )
    decoder_model.eval()

    # Sample initial states from dataset
    logging.info(f"Sampling {args.n_rollouts} initial states from dataset")
    n_samples = min(args.n_rollouts, len(df))
    sample_indices = np.random.choice(len(df), size=n_samples, replace=False)

    # Perform rollouts
    rngs = nnx.Rngs(args.seed)
    all_trajectories = []

    for i, idx in enumerate(tqdm(sample_indices)):
        logging.info(f"Rollout {i + 1}/{n_samples}")

        # Get initial embedding (convert numpy int to python int for polars)
        initial_embedding = jnp.array(df["embedding"][int(idx)])

        # Perform rollout
        trajectory = autoregressive_rollout(
            transition_model,
            decoder_model,
            initial_embedding,
            args.policy,
            df,
            args.rollout_strategy,
            args.K,
            rngs,
        )

        all_trajectories.append(trajectory)

    # Find nearest images for each embedding in trajectories
    logging.info("Finding nearest images for each embedding...")
    trajectories_with_images = []

    for traj_idx, trajectory in enumerate(tqdm(all_trajectories)):
        traj_with_images = []
        for step_idx, step in enumerate(tqdm(trajectory)):
            image_path, similarity, area = find_nearest_image(step["embedding"], df)
            traj_with_images.append(
                {
                    **step,
                    "image_path": image_path,
                    "similarity": similarity,
                    "dataset_area": area,
                }
            )
            logging.info(
                f"Trajectory {traj_idx + 1}, Step {step_idx + 1}: "
                f"Area={step['area']:.1f}, Similarity={similarity:.3f}"
            )
        trajectories_with_images.append(traj_with_images)

    # Plot trajectories
    logging.info("Plotting trajectories...")
    plot_trajectories(
        trajectories_with_images,
        data_dir=Path("/data/plant-rl/offline"),
        output_dir=output_dir,
        policy_name=args.policy,
    )

    logging.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
