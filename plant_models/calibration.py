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
        max_state_dist: float = 0.1,
        max_action_dist: float = 0.1,
        render_mode: str | None = None,
    ):
        self.render_mode = render_mode
        self.dataset = minari.load_dataset(dataset_id)
        self.k = k
        self.max_state_dist = max_state_dist
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
        self.X_action_np = np.concatenate(actions, axis=0)
        self.X_next_state_np = np.concatenate(next_observations, axis=0)
        self.rewards_np = np.concatenate(rewards, axis=0)
        self.terminals_np = np.concatenate(terminals, axis=0)
        self.truncateds_np = np.concatenate(truncateds, axis=0)
        self.returns_np = np.array(returns)

        self.default_return = np.min(self.returns_np)

        self.X_state = jnp.array(self.X_state_np)
        self.X_action = jnp.array(self.X_action_np)

        # Empirical stats
        self.norm_mean = np.mean(self.X_state_np, axis=0)
        self.norm_std = np.std(self.X_state_np, axis=0)

        # Compute L2 Norms for Cosine Similarity
        X_state_norm = (self.X_state_np - self.norm_mean) / np.clip(
            self.norm_std, min=1e-8
        )
        X_state_norm = jnp.array(X_state_norm)

        # L2 Normalize for Cosine Index
        norm_state = jnp.linalg.norm(X_state_norm, axis=1, keepdims=True)
        self.X_state_emb = X_state_norm / jnp.clip(norm_state, min=1e-8)

        norm_action = jnp.linalg.norm(self.X_action, axis=1, keepdims=True)
        self.X_action_emb = self.X_action / jnp.clip(norm_action, min=1e-8)

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
        self.seen_mask = jnp.zeros(self.X_state.shape[0], dtype=bool)
        self.seen_mask = self.seen_mask.at[dataset_idx].set(True)

        self.current_return = 0.0

        self.current_image_path = self.image_paths[dataset_idx]

        return self.current_state, {
            "area": self.area[dataset_idx],
            "image_path": self.image_paths[dataset_idx],
        }

    def step(self, action):
        # Normalize State
        self.norm_state = (self.current_state - self.norm_mean) / self.norm_std

        # L2 Normalize Query State
        q_state = jnp.array(self.norm_state)
        q_state_emb = q_state / jnp.clip(jnp.linalg.norm(q_state), min=1e-8)

        # L2 Normalize Query Action
        q_action = jnp.array(action)
        q_action_emb = q_action / jnp.clip(jnp.linalg.norm(q_action), min=1e-8)

        # Run Search
        self.key, subkey = jax.random.split(self.key)

        idx, self.seen_mask, best_state_dist, best_action_dist = self._find_neighbor(
            q_state_emb,
            q_action_emb,
            self.seen_mask,
            self.X_state_emb,
            self.X_action_emb,
            subkey,
        )

        if idx < 0:
            # Terminate if no valid neighbors, adjust reward so return is default_return
            error_codes = {
                -1: "no_neighbors",
                -2: f"state_threshold, best_state_dist: {best_state_dist}",
                -3: f"action_threshold, best_action_dist: {best_action_dist}",
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
                    "area": self.area[idx],
                    "image_path": self.image_paths[idx],
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
        q_state: jax.Array,
        q_action: jax.Array,
        seen_mask: jax.Array,
        X_state_emb: jax.Array,
        X_action_emb: jax.Array,
        key: jax.Array,
    ):
        sim_state = X_state_emb @ q_state
        sim_action = X_action_emb @ q_action

        scores = sim_state * sim_action

        # Mask out seen states
        scores = jnp.where(seen_mask, -jnp.inf, scores)

        # Top K
        top_k_scores, top_k_indices = jax.lax.top_k(scores, self.k)

        best_idx = top_k_indices[0]
        best_state_dist = 1 - sim_state[best_idx]
        best_action_dist = 1 - sim_action[best_idx]

        found = top_k_scores[0] > -1e9
        state_ok = best_state_dist <= self.max_state_dist
        action_ok = best_action_dist <= self.max_action_dist

        failure_code = jax.lax.select(
            state_ok,
            -3,
            -2,
        )
        failure_code = jax.lax.select(
            found,
            failure_code,
            -1,
        )

        valid_neighbor = found & state_ok & action_ok

        probs = jax.nn.softmax(top_k_scores)
        choice_idx = jax.random.choice(key, top_k_indices, p=probs)
        new_mask = seen_mask.at[choice_idx].set(True)

        final_idx = jax.lax.select(valid_neighbor, choice_idx, failure_code)

        return final_idx, new_mask, best_state_dist, best_action_dist
