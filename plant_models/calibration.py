import functools
import math

import gymnasium as gym
import jax
import jax.numpy as jnp
import minari
import numpy as np
from PIL import Image


class PlantCalibrationModel(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        dataset_id: str = "plant-data/mixed-v19",
        k: int = 3,
        max_stat_dist: float = 0.1,
        max_emb_dist: float = 0.1,
        max_action_dist: float = 0.1,
        render_mode: str | None = None,
    ):
        self.render_mode = render_mode
        self.dataset = minari.load_dataset(dataset_id)
        self.k = k
        self.max_stat_dist = max_stat_dist
        self.max_emb_dist = max_emb_dist
        self.max_action_dist = max_action_dist
        self.observation_space = self.dataset.observation_space
        self.action_space = self.dataset.action_space
        self.key = jax.random.key(0)
        self._load_dataset()

    def _load_dataset(self):
        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminals = []
        truncateds = []
        returns = []

        self.initial_indices = []
        current_idx = 0

        image_paths = []

        for episode in self.dataset:
            obs = episode.observations
            # wall_time == 0.0 and area > 0.0
            if math.isclose(obs[0, 0], 0.0) and obs[0, 1] > 0.0:
                self.initial_indices.append(current_idx)
            acts = episode.actions
            rews = episode.rewards
            terms = episode.terminations
            truncs = episode.truncations

            imgs = episode.infos.get("image_path", [None] * len(obs))
            if len(imgs) != len(obs):
                imgs = [None] * len(obs)

            T = len(acts)
            observations.append(obs[:-1])
            next_observations.append(obs[1:])
            actions.append(acts)
            rewards.append(rews)
            terminals.append(terms)
            truncateds.append(truncs)
            image_paths.append(imgs[:-1])

            current_idx += T
            returns.append(np.sum(rews))

        area = [obs[:, 1] for obs in observations]

        self.area = np.concatenate(area, axis=0)
        self.image_paths = np.concatenate(image_paths, axis=0)
        self.X_state_np = np.concatenate(observations, axis=0)

        # Split State
        # Last 768 are embedding
        self.X_stat_np = self.X_state_np[:, :-768]
        self.X_emb_np = self.X_state_np[:, -768:]

        self.X_action_np = np.concatenate(actions, axis=0)
        self.X_next_state_np = np.concatenate(next_observations, axis=0)
        self.rewards_np = np.concatenate(rewards, axis=0)
        self.terminals_np = np.concatenate(terminals, axis=0)
        self.truncateds_np = np.concatenate(truncateds, axis=0)
        self.returns_np = np.array(returns)

        self.default_return = np.min(self.returns_np)

        self.X_stat = jnp.array(self.X_stat_np)
        self.X_emb = jnp.array(self.X_emb_np)
        self.X_action = jnp.array(self.X_action_np)

        # Empirical stats for Plant Stats (Euclidean)
        self.norm_mean = np.mean(self.X_stat_np, axis=0)
        self.norm_std = np.std(self.X_stat_np, axis=0)

        # Z-Score Normalize Stats
        X_stat_norm = (self.X_stat_np - self.norm_mean) / np.clip(
            self.norm_std, min=1e-8
        )
        self.X_stat_norm = jnp.array(X_stat_norm)

        # L2 Normalize Embeddings (Cosine)
        norm_emb = jnp.linalg.norm(self.X_emb, axis=1, keepdims=True)
        self.X_emb_norm = self.X_emb / jnp.clip(norm_emb, min=1e-8)

        # No normalization for Action, we use raw Euclidean distance on the simplex
        # (Assuming actions are already somewhat normalized or on a simplex)
        self.X_action_jax = self.X_action

        self.seen_mask = jnp.zeros(current_idx, dtype=bool)

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.key = jax.random.key(seed)

        self.key, subkey = jax.random.split(self.key)

        # Pick random initial index
        idx = int(jax.random.randint(subkey, (1,), 0, len(self.initial_indices))[0])
        dataset_idx = self.initial_indices[idx]

        self.current_state = np.array(self.X_state_np[dataset_idx])

        # Reset seen mask
        self.seen_mask = jnp.zeros(self.X_stat.shape[0], dtype=bool)
        self.seen_mask = self.seen_mask.at[dataset_idx].set(True)

        self.current_return = 0.0

        self.current_image_path = self.image_paths[dataset_idx]

        return self.current_state, {
            "area": self.area[dataset_idx],
            "image_path": self.image_paths[dataset_idx],
        }

    def step(self, action):
        # Normalize Query Stat
        q_stat = self.current_state[:-768]
        q_emb = self.current_state[-768:]

        q_stat_norm = (q_stat - self.norm_mean) / np.clip(self.norm_std, min=1e-8)
        q_stat_norm = jnp.array(q_stat_norm)

        # L2 Normalize Query Embedding
        q_emb_jax = jnp.array(q_emb)
        q_emb_norm = q_emb_jax / jnp.clip(jnp.linalg.norm(q_emb_jax), min=1e-8)

        # Query Action (Raw)
        q_action = jnp.array(action)

        # Run Search
        self.key, subkey = jax.random.split(self.key)

        idx, self.seen_mask, best_stat_dist, best_emb_dist, best_action_dist = (
            self._find_neighbor(
                q_stat_norm,
                q_emb_norm,
                q_action,
                self.seen_mask,
                self.X_stat_norm,
                self.X_emb_norm,
                self.X_action_jax,
                subkey,
            )
        )

        if idx < 0:
            # Terminate if no valid neighbors, adjust reward so return is default_return
            error_codes = {
                -1: "no_neighbors",
                -2: f"stat_threshold: {best_stat_dist:.2f}",
                -3: f"emb_threshold: {best_emb_dist:.2f}",
                -4: f"action_threshold: {best_action_dist:.2f}",
            }
            error_msg = error_codes.get(int(idx), "unknown_error")

            reward = self.default_return - self.current_return
            self.current_return += reward

            self.current_image_path = self.image_paths[idx]

            return (
                self.current_state,
                reward,
                True,
                False,
                {
                    "error": error_msg,
                    "area": None,
                    "image_path": None,
                },
            )

        idx = int(idx)

        # Retrieve Transitions
        next_state = self.X_next_state_np[idx]
        reward = self.rewards_np[idx]
        terminated = self.terminals_np[idx]
        truncated = self.truncateds_np[idx]

        self.current_state = np.array(next_state)
        reward = float(reward)
        terminated = bool(terminated)
        # Ignore truncated flag to allow stitching trajectories
        truncated = False
        terminated = bool(terminated)

        self.current_return += reward

        self.current_image_path = self.image_paths[idx]

        return (
            self.current_state,
            reward,
            terminated,
            truncated,
            {
                "area": self.area[idx],
                "image_path": self.image_paths[idx],
            },
        )

    def render(self):
        if self.render_mode == "rgb_array":
            if (
                hasattr(self, "current_image_path")
                and self.current_image_path is not None
            ):
                path = self.current_image_path
                if isinstance(path, (bytes, np.bytes_)):
                    path = path.decode("utf-8")

                try:
                    img = Image.open(path)
                    img = img.resize((224, 224))
                    return np.array(img)
                except Exception:
                    return None
        return None

    @functools.partial(jax.jit, static_argnums=(0,))
    def _find_neighbor(
        self,
        q_stat: jax.Array,
        q_emb: jax.Array,
        q_action: jax.Array,
        seen_mask: jax.Array,
        X_stat_norm: jax.Array,
        X_emb_norm: jax.Array,
        X_action: jax.Array,
        key: jax.Array,
    ):
        # 1. Stat Distance (Euclidean on Z-scored stats)
        # ||x - q||^2 = ||x||^2 + ||q||^2 - 2 <x, q>
        # but simpler to just compute diff since dimension is small
        diff_stat = X_stat_norm - q_stat
        dist_stat = jnp.sqrt(jnp.sum(diff_stat**2, axis=1))

        # 2. Embedding Distance (Cosine)
        # 1 - <x, q> (since vectors are L2 normalized)
        sim_emb = X_emb_norm @ q_emb
        dist_emb = 1.0 - sim_emb
        # Clip to avoid negative due to precision
        dist_emb = jnp.maximum(dist_emb, 0.0)

        # 3. Action Distance (Euclidean)
        diff_action = X_action - q_action
        dist_action = jnp.sqrt(jnp.sum(diff_action**2, axis=1))

        # Total "Cost" (Sum of distances)
        # Minimize Cost <-> Maximize Score
        total_dist = dist_stat + dist_emb + dist_action
        scores = -total_dist

        # Mask out seen states
        scores = jnp.where(seen_mask, -jnp.inf, scores)

        # Top K
        top_k_scores, top_k_indices = jax.lax.top_k(scores, self.k)

        best_idx = top_k_indices[0]

        # Recalculate best distances for the chosen candidate to return
        best_stat_dist = dist_stat[best_idx]
        best_emb_dist = dist_emb[best_idx]
        best_action_dist = dist_action[best_idx]

        found = top_k_scores[0] > -1e9

        stat_ok = best_stat_dist <= self.max_stat_dist
        emb_ok = best_emb_dist <= self.max_emb_dist
        action_ok = best_action_dist <= self.max_action_dist

        # Failure Checks
        # Priority: Action > Emb > Stat
        failure_code = -1  # No neighbors found

        # If found, check thresholds
        # If action fails -> -4
        # Else if emb fails -> -3
        # Else if stat fails -> -2
        # Else -> valid index

        failure_code = jax.lax.select(
            action_ok, -2, -4
        )  # If action ok, maybe stat failed (-2). If not, -4.
        failure_code = jax.lax.select(
            emb_ok, failure_code, -3
        )  # If emb ok, keep previous. If not, -3.

        # If everything ok, we use best_idx. If any failed, we use failure_code.
        neighbor_valid = stat_ok & emb_ok & action_ok

        # If not found at all, stays -1.
        final_validity = found & neighbor_valid

        # Softmax sampling from top K
        probs = jax.nn.softmax(top_k_scores)
        choice_idx = jax.random.choice(key, top_k_indices, p=probs)
        new_mask = seen_mask.at[choice_idx].set(True)

        final_idx = jax.lax.select(final_validity, choice_idx, failure_code)

        # Only return -1 if truly nothing found (mask full or similar)
        final_idx = jax.lax.select(found, final_idx, -1)

        return final_idx, new_mask, best_stat_dist, best_emb_dist, best_action_dist
