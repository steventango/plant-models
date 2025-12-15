import gymnasium as gym
import jax

from PlantCalibrationModel import PlantCalibrationModel


def test_initialization():
    """Test that the environment initializes and loads the dataset."""
    env = PlantCalibrationModel(dataset_id="plant-data/mixed-v19", k=3)
    assert isinstance(env, gym.Env)
    assert env.observation_space is not None
    assert env.action_space is not None


def test_reset():
    """Test the reset functionality."""
    env = PlantCalibrationModel(dataset_id="plant-data/mixed-v19", k=3)
    obs, info = env.reset(seed=42)

    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)
    assert "area" in info
    assert "image_path" in info


def test_step():
    """Test a single step in the environment."""
    env = PlantCalibrationModel(dataset_id="plant-data/mixed-v19", k=3)
    obs, info = env.reset(seed=0)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "area" in info
    assert "image_path" in info


def test_rollout():
    """Check that we can run a rollout."""
    env = PlantCalibrationModel(dataset_id="plant-data/mixed-v19", k=3)
    obs, _ = env.reset(seed=123)

    for _ in range(14):
        action = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(action)
        if term or trunc:
            break


def test_stitching():
    """Test that the environment continues (doesn't truncate) when hitting a truncated state."""
    env = PlantCalibrationModel(dataset_id="plant-data/mixed-v19", k=3)
    env.reset(seed=42)

    env.truncateds_np[:] = True

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert truncated is False


def test_no_neighbor_termination():
    """Test that the env terminates with default return if no neighbor is found."""
    env = PlantCalibrationModel(dataset_id="plant-data/mixed-v19", k=3)
    env.reset(seed=42)

    action = env.action_space.sample()
    action[:] = 0.0

    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert reward == env.default_return


def test_error_no_neighbors():
    """Test 'no_neighbors' error code."""
    env = PlantCalibrationModel(dataset_id="plant-data/mixed-v19")
    env.reset(seed=42)

    env.seen_mask = jax.numpy.ones(env.seen_mask.shape[0], dtype=bool)

    action = env.action_space.sample()
    _, _, terminated, _, info = env.step(action)

    assert terminated
    assert info["error"] == "no_neighbors"


def test_error_state_threshold():
    """Test 'state_threshold' error code."""
    # Strict state, loose action
    env = PlantCalibrationModel(
        dataset_id="plant-data/mixed-v19", max_state_dist=-0.1, max_action_dist=10.0
    )
    env.reset(seed=42)

    action = env.action_space.sample()
    _, _, terminated, _, info = env.step(action)

    assert terminated
    assert info["error"].startswith("state_threshold")


def test_error_action_threshold():
    """Test 'action_threshold' error code."""
    # Loose state, strict action
    env = PlantCalibrationModel(
        dataset_id="plant-data/mixed-v19", max_state_dist=10.0, max_action_dist=-0.1
    )
    env.reset(seed=42)

    action = env.action_space.sample()
    _, _, terminated, _, info = env.step(action)

    assert terminated
    assert info["error"].startswith("action_threshold")


def test_threshold_termination():
    """Test that the env terminates if the best neighbor violates thresholds."""
    env = PlantCalibrationModel(
        dataset_id="plant-data/mixed-v19", k=3, max_state_dist=0.0, max_action_dist=0.0
    )
    env.reset(seed=42)

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert reward == env.default_return
