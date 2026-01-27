import math
import numpy as np
import pytest
from plant_models.calibration import PlantGrowthChamberModel

N_PLANTS = 64


@pytest.fixture
def growth_chamber():
    """Fixture to initialize the PlantGrowthChamberModel."""
    return PlantGrowthChamberModel(
        n_plants=N_PLANTS,
        dataset_id="plant-data/mixed-all-v20",
        k=10,
        max_stat_dist=3.0,
        max_emb_dist=1.0,
        render_mode="rgb_array",
    )


def test_reset(growth_chamber):
    """Test the reset method of the growth chamber."""
    obs, info = growth_chamber.reset(seed=42)

    assert obs.ndim == 1
    assert obs.shape[0] > 768

    assert "infos" in info
    assert len(info["infos"]) == N_PLANTS


def test_step(growth_chamber):
    """Test the step method of the growth chamber."""
    growth_chamber.reset(seed=42)
    action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = growth_chamber.step(action)

    assert next_obs.shape == growth_chamber.observation_space.shape
    assert isinstance(reward, (float, np.float32, np.float64))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "infos" in info
    assert len(info["infos"]) == N_PLANTS


def test_observation_modes():
    """Test mean and median observation modes by initializing separate environments."""
    kwargs = {
        "n_plants": N_PLANTS,
        "dataset_id": "plant-data/mixed-all-v20",
        "k": 10,
        "max_stat_dist": 3.0,
        "max_emb_dist": 1.0,
    }

    # Test median
    env_median = PlantGrowthChamberModel(obs_mode="median", **kwargs)
    obs_median, _ = env_median.reset(seed=42)

    # Test mean
    env_mean = PlantGrowthChamberModel(obs_mode="mean", **kwargs)
    obs_mean, _ = env_mean.reset(seed=42)

    # With seed 42 and N=4, they should be different
    diff = np.abs(obs_median - obs_mean).sum()
    assert diff > 0, (
        f"Mean and median observations should be different for N={N_PLANTS}"
    )


def test_grid_rendering(growth_chamber):
    """Test the grid rendering functionality."""
    growth_chamber.reset(seed=42)
    grid = growth_chamber.render()

    assert grid is not None
    assert grid.ndim == 3
    w = math.ceil(math.sqrt(N_PLANTS))
    h = math.ceil(N_PLANTS / w)
    assert grid.shape == (224 * h, 224 * w, 3)


def test_termination_and_truncation():
    """Test that the chamber terminates on step limit or all plants done."""
    n_plants = N_PLANTS
    chamber_limit = 5
    env = PlantGrowthChamberModel(
        n_plants=n_plants,
        dataset_id="plant-data/mixed-all-v20",
        terminal_episode_steps=chamber_limit,
        k=5,
        max_stat_dist=5.0,
    )

    env.reset(seed=42)

    for step in range(1, chamber_limit + 1):
        action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        _, reward, terminated, truncated, _ = env.step(action)

        assert isinstance(reward, (float, np.float32, np.float64))
        if step < chamber_limit:
            assert not terminated
            assert not truncated
        else:
            # At step 5, it should be truncated if not all plants terminated
            assert truncated or terminated

    assert env.current_episode_steps == chamber_limit
