import json
import logging
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationModel:
    def __init__(
        self,
        data_path: str,
        normalization_path: str,
        state_cols: list[str],
        action_cols: list[str],
        K: int = 5,
        state_threshold: float = 0.9,
        action_threshold: float = 0.9,
        seed: int = 0,
    ):
        """
        kNN Calibration Model using Cosine Similarity.

        Args:
            data_path: Path to the parquet dataset.
            normalization_path: Path to the normalization stats json.
            state_cols: List of columns to use as state.
            action_cols: List of columns to use as action.
            K: Number of neighbors to retrieve.
            state_threshold: Minimum similarity for state match.
            action_threshold: Minimum similarity for action match.
            seed: Random seed for reproducibility.
        """
        self.data_path = Path(data_path)
        self.normalization_path = Path(normalization_path)
        self.state_cols = state_cols
        self.action_cols = action_cols
        self.K = K
        self.state_threshold = state_threshold
        self.action_threshold = action_threshold
        self.seed = seed
        np.random.seed(seed)

        # Load normalization stats
        with open(self.normalization_path, "r") as f:
            self.norm_stats = json.load(f)

        logger.info(f"Loading data from {self.data_path}")
        self.df = pl.read_parquet(self.data_path)

        # Preprocess and build the transition dataset
        self._build_dataset()

    def _get_normalized_values(self, df: pl.DataFrame, cols: list[str]) -> np.ndarray:
        """Extracts and normalizes columns from dataframe."""
        vals_list = []
        for col in cols:
            # Check if column is a list/array (embedding)
            if df[col].dtype in [pl.List, pl.Array]:
                # Assume embedding columns don't use min-max normalization from JSON
                # but we will L2 normalize them later for cosine similarity
                # Convert to numpy matrix
                mat = np.stack(df[col].to_numpy())
                mat = np.nan_to_num(mat, nan=0.0)
                vals_list.append(mat)
            else:
                # Scalar column
                vals = df[col].to_numpy()
                vals = np.nan_to_num(vals, nan=0.0)
                if col in self.norm_stats:
                    min_val = self.norm_stats[col]["min"]
                    max_val = self.norm_stats[col]["max"]
                    if max_val > min_val:
                        vals = (vals - min_val) / (max_val - min_val)
                    else:
                        vals = np.zeros_like(vals)

                vals_list.append(vals.reshape(-1, 1))

        if not vals_list:
            raise ValueError(f"No valid columns found in {cols}")

        return np.concatenate(vals_list, axis=1)

    def _build_dataset(self):
        df = self.df.sort(["experiment", "zone", "plant_id", "time"])
        df_next = df.shift(-1)

        # Valid transitions: same experiment, zone, plant_id
        valid_mask = (
            (df["experiment"] == df_next["experiment"])
            & (df["zone"] == df_next["zone"])
            & (df["plant_id"] == df_next["plant_id"])
            & (df["time"] < df_next["time"])
        )

        # Filter valid transitions
        self.transitions = df.filter(valid_mask)
        self.next_transitions = df_next.filter(valid_mask)

        logger.info(f"Found {len(self.transitions)} valid transitions.")

        # Prepare Search Index (States and Actions)
        # We need numpy arrays for JAX

        # 1. State Vectors
        X_state = self._get_normalized_values(self.transitions, self.state_cols)
        self.X_state = jax.device_put(jnp.array(X_state))

        # 2. Action Vectors
        X_action = self._get_normalized_values(self.transitions, self.action_cols)
        self.X_action = jax.device_put(jnp.array(X_action))

        # Normalize vectors for Cosine Similarity (L2 norm)
        # Handle zero vectors to avoid NaN
        self.X_state_norm = self.X_state / jnp.clip(
            jnp.linalg.norm(self.X_state, axis=1, keepdims=True), a_min=1e-8
        )
        self.X_action_norm = self.X_action / jnp.clip(
            jnp.linalg.norm(self.X_action, axis=1, keepdims=True), a_min=1e-8
        )

        self.initial_transitions = self.transitions.filter(pl.col("wall_time") == 0)
        logger.info(
            f"Found {len(self.initial_transitions)} valid initial transitions (wall_time=0)."
        )

        # TODO: move to plant-data
        returns = (
            self.transitions.filter(pl.col("terminal").cast(pl.Boolean))
            .filter(
                pl.col("wall_time").over(["experiment", "zone", "plant_id"]).max() >= 13
            )
            .group_by(["experiment", "zone", "plant_id"])
            .agg(pl.col("reward").sum())
            .sort("reward", descending=True)
        )
        self.default_return = returns["reward"][0]

    @property
    def num_samples(self):
        return self.X_state.shape[0]

    def sample_initial_state(self, seed: int = 0):
        """Samples a random initial state from the dataset."""
        np.random.seed(seed)
        idx = np.random.randint(0, len(self.initial_transitions))
        row = self.initial_transitions[idx]

        state = {}
        for col in self.state_cols:
            state[col] = row[col][0]
        seen = {idx}
        return state, seen

    def normalize_input(self, data: dict, cols: list[str]) -> np.ndarray:
        """normalize single input dict to vector"""
        vals_list = []
        for col in cols:
            val = data[col]
            # Ensure val is numpy array
            arr = np.array(val)
            arr = np.nan_to_num(arr, nan=0.0)
            # Flatten to 1D
            vals_list.append(arr.reshape(-1))

            # Note: For scalars we handle normalization logic if scalar
            if col in self.norm_stats and arr.size == 1:
                # Check if we need to normalize this single value
                # Using the previously flattened array which is now 1D size 1
                v = vals_list[-1][0]
                min_val = self.norm_stats[col]["min"]
                max_val = self.norm_stats[col]["max"]
                if max_val > min_val:
                    v = (v - min_val) / (max_val - min_val)
                else:
                    v = 0.0
                vals_list[-1][0] = v

        return np.concatenate(vals_list)

    def predict(self, state: dict, action: dict, seen: set[int]):
        """
        Predict next state, reward, terminal.

        Returns:
            next_state (dict), reward (float), terminal (bool)
        """
        # Vectorize input
        q_state = self.normalize_input(state, self.state_cols)
        q_action = self.normalize_input(action, self.action_cols)

        q_state = jax.device_put(jnp.array(q_state))
        q_action = jax.device_put(jnp.array(q_action))

        # Normalize query
        q_state_norm = q_state / jnp.clip(jnp.linalg.norm(q_state), min=1e-8)
        q_action_norm = q_action / jnp.clip(jnp.linalg.norm(q_action), min=1e-8)

        # Compute Similarities
        sim_state = jnp.dot(self.X_state_norm, q_state_norm)
        sim_action = jnp.dot(self.X_action_norm, q_action_norm)

        # Combine Score
        scores = sim_state * sim_action

        # Top K
        top_k_scores, top_k_indices = jax.lax.top_k(scores, self.K)
        top_k_scores = np.array(top_k_scores)
        top_k_indices = np.array(top_k_indices)

        # drop seen indices from top_k_indices
        top_k_indices_filtered = []
        top_k_scores_filtered = []
        for idx, score in zip(top_k_indices, top_k_scores):
            if idx not in seen:
                top_k_indices_filtered.append(idx)
                top_k_scores_filtered.append(score)

        top_k_indices = np.array(top_k_indices_filtered)
        top_k_scores = np.array(top_k_scores_filtered)

        if not top_k_indices.size:
            logger.info("No valid matches found. Using default return.")
            return None, None, True, {}, seen

        best_score = top_k_scores[0]

        best_idx = int(top_k_indices[0])
        best_state_sim = float(sim_state[best_idx])
        best_action_sim = float(sim_action[best_idx])

        logger.info(
            f"Best Match stats: Score={best_score:.4f}, StateSim={best_state_sim:.4f}, ActionSim={best_action_sim:.4f}"
        )

        if best_state_sim < self.state_threshold:
            logger.info(
                f"State similarity {best_state_sim:.4f} below threshold {self.state_threshold}. Terminating."
            )
            return None, None, True, {}, seen

        if best_action_sim < self.action_threshold:
            logger.info(
                f"Action similarity {best_action_sim:.4f} below threshold {self.action_threshold}. Terminating."
            )
            return None, None, True, {}, seen

        # log top K
        for i in range(top_k_indices.size):
            idx = top_k_indices[i]
            row = self.transitions[int(idx)]
            next_row = self.next_transitions[int(idx)]
            logger.info(
                f"{i}: Score={top_k_scores[i]:.4f}, StateSim={sim_state[idx]:.4f}, ActionSim={sim_action[idx]:.4f}"
            )
            logger.info(
                f"  Curr: Idx={idx}, Exp={row['experiment'][0]}, Zone={row['zone'][0]}, Time={row['time'][0]}"
            )
            logger.info(
                f"  Next: Idx={idx + 1}, Exp={next_row['experiment'][0]}, Zone={next_row['zone'][0]}, Time={next_row['time'][0]}"
            )

        # Handle potential NaNs in scores (safe-guard)
        top_k_scores = np.nan_to_num(top_k_scores, nan=-1.0)

        # Sampling
        probs = jax.nn.softmax(top_k_scores)
        probs = np.array(probs)
        neighbor_idx = np.random.choice(top_k_indices, p=probs)
        
        # Mark as seen
        seen.add(int(neighbor_idx))

        # Retrieve next state info
        row = self.next_transitions[int(neighbor_idx)]

        # Print the selected row idx
        logger.info(f"Selected row idx: {neighbor_idx}")

        # Construct result dict
        next_state = {}
        for col in self.state_cols:
            if col in row:
                val = row[col][0]
                # If it's a list (embedding), keep as list or array?
                # The user code seems to expect lists/arrays suitable for next query.
                next_state[col] = val

        reward = row["reward"][0] if "reward" in row else 0.0
        terminal = row["terminal"][0] if "terminal" in row else False

        info = {
            "image_path": row["image_path"][0] if "image_path" in row else None,
            "experiment": row["experiment"][0] if "experiment" in row else None,
            "zone": row["zone"][0] if "zone" in row else None,
            "time": row["time"][0] if "time" in row else None,
        }

        return next_state, reward, terminal, info, seen
