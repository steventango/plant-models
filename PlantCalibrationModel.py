import functools

import gymnasium as gym
import jax
import jax.numpy as jnp
import minari
import numpy as np


class PlantCalibrationModel(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        dataset_id: str = "plant-data/mixed-v18",
        k: int = 3,
        max_state_dist: float = 0.1,
        max_action_dist: float = 0.1,
    ):
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

        for episode in self.dataset:
            self.initial_indices.append(current_idx)
            obs = episode.observations
            acts = episode.actions
            rews = episode.rewards
            terms = episode.terminations
            truncs = episode.truncations
            T = len(acts)
            observations.append(obs[:-1])
            next_observations.append(obs[1:])
            actions.append(acts)
            rewards.append(rews)
            terminals.append(terms)
            truncateds.append(truncs)
            current_idx += T
            returns.append(np.sum(rews))

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

        return self.current_state, {}

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

        idx, self.seen_mask = self._find_neighbor(
            q_state_emb,
            q_action_emb,
            self.seen_mask,
            self.X_state_emb,
            self.X_action_emb,
            subkey,
        )

        if idx == -1:
            # Terminate if no valid neighbors, adjust reward so return is default_return
            reward = self.default_return - self.current_return
            self.current_return += reward
            return self.current_state, reward, True, False, {"error": "no_neighbors"}

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

        return self.current_state, reward, terminated, truncated, {}

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

        # Valid if score > -inf AND distances <= thresholds
        valid_neighbor = (
            (top_k_scores[0] > -1e9)
            & (best_state_dist <= self.max_state_dist)
            & (best_action_dist <= self.max_action_dist)
        )

        probs = jax.nn.softmax(top_k_scores)
        choice_idx = jax.random.choice(key, top_k_indices, p=probs)
        new_mask = seen_mask.at[choice_idx].set(True)

        final_idx = jax.lax.select(valid_neighbor, choice_idx, -1)

        return final_idx, new_mask
