#!/bin/bash

# Run autoregressive rollouts with different policies

echo "Running autoregressive rollouts..."

K=21
n_rollouts=1
seed=2

# White light policy (as requested)
# echo "Running white light policy..."
# uv run python rollout.py --K $K --n_rollouts $n_rollouts --policy white --output_dir results/rollout_white

echo "Running red light policy..."
uv run python rollout.py --K $K --n_rollouts $n_rollouts --policy red --output_dir results/rollout_red --seed $seed
# uv run python rollout.py --K $K --n_rollouts $n_rollouts --rollout_strategy k_nearest --policy red --output_dir results/rollout_red_k_nearest --seed $seed

echo "Running blue light policy..."
uv run python rollout.py --K $K --n_rollouts $n_rollouts --policy blue --output_dir results/rollout_blue --seed $seed
# uv run python rollout.py --K $K --n_rollouts $n_rollouts --rollout_strategy k_nearest --policy blue --output_dir results/rollout_blue_k_nearest --seed $seed

# echo "Running random policy..."
# uv run python rollout.py --K $K --n_rollouts $n_rollouts --policy random --output_dir results/rollout_random

echo "Done! Check results/rollout_* directories for outputs."
