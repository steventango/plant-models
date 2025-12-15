import gymnasium as gym
import jax

from PlantCalibrationModel import PlantCalibrationModel


def test_initialization():
    """Test that the environment initializes and loads the dataset."""
    env = PlantCalibrationModel(dataset_id="plant-data/mixed-v18", k=3)
    assert isinstance(env, gym.Env)
    assert env.observation_space is not None
    assert env.action_space is not None

    # Check if JAX arrays are built
    assert hasattr(env, "X_state")
    assert hasattr(env, "X_action")
    # Should be JAX Arrays
    assert isinstance(env.X_state, jax.Array)


def test_reset():
    """Test the reset functionality."""
    env = PlantCalibrationModel(dataset_id="plant-data/mixed-v18", k=3)
    obs, info = env.reset(seed=42)

    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)


def test_step():
    """Test a single step in the environment."""
    env = PlantCalibrationModel(dataset_id="plant-data/mixed-v18", k=3)
    obs, info = env.reset(seed=0)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_rollout():
    """Check that we can run a rollout."""
    env = PlantCalibrationModel(dataset_id="plant-data/mixed-v18", k=3)
    obs, _ = env.reset(seed=123)

    for _ in range(14):
        action = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(action)
        if term or trunc:
            break


def test_no_neighbor_termination():
    """Test that the env terminates with default return if no neighbor is found."""
    env = PlantCalibrationModel(dataset_id="plant-data/mixed-v18", k=3)
    env.reset(seed=42)

    action = env.action_space.sample()
    action[:] = 0.0

    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert reward == env.default_return


def test_threshold_termination():
    """Test that the env terminates if the best neighbor violates thresholds."""
    env = PlantCalibrationModel(
        dataset_id="plant-data/mixed-v18", k=3, max_state_dist=0.0, max_action_dist=0.0
    )
    env.reset(seed=42)

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert reward == env.default_return
