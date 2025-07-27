from itertools import product
import numpy as np
import hashlib
import gymnasium as gym
import gymnasium.spaces as spaces

def normalise_reward(reward, min, max):
    normalised_reward = (reward-min)/(max-min)
    return normalised_reward

def sample_hyperparams(num_configs):
    np.random.seed(0)
    hyperparameters = []
    for _ in range(num_configs):
        # Learning rate: [0,1] with higher probability in (0,0.1)
        lr = np.random.beta(0.5, 3.0)  # Strongly skewed toward 0
        
        # Gamma: [0,1] with higher probability in (0.9,1)
        gamma = np.random.beta(5.0, 0.5)  # Strongly skewed toward 1
        
        # Epsilon: [0,1] with higher probability in (0,0.3)
        epsilon = np.random.beta(0.8, 3.0)  # Moderately skewed toward 0
        
        hyperparameters.append(np.array([np.round(lr, decimals=3), np.round(gamma, decimals=3), np.round(epsilon, decimals=3)]))
    
    return hyperparameters

def sample_hyperparams_pg(num_configs):
    hyperparameters = []
    for _ in range(num_configs):
        # Learning rate: [0,1] with higher probability in (0,0.1)
        lr = np.random.beta(0.5, 3.0)  # Strongly skewed toward 0
        
        # Gamma: [0,1] with higher probability in (0.9,1)
        gamma = np.random.beta(5.0, 0.5)  # Strongly skewed toward 1
        
        # Epsilon: [0,1] with higher probability in (0,0.4)
        epsilon = np.random.beta(1.0, 2.0)  # Moderately skewed toward 0
        
        hyperparameters.append(np.array([np.round(lr, decimals=3), np.round(gamma, decimals=3), np.round(epsilon, decimals=3)]))
    
    return hyperparameters

def avg_reward(rewards):
    return np.mean(rewards)

# Predefined mappings for categorical features
OBS_SPACE_TYPES = {"Box": 0, "Discrete": 1, "Dict": 2, "MultiBinary": 3, "MultiDiscrete": 4}
ACTION_SPACE_TYPES = {"Discrete": 0, "Box": 1, "MultiDiscrete": 2}
DTYPE_MAP = {"float32": 0, "float64": 1, "int32": 2, "int64": 3, "uint8": 4}
REWARD_DENSITY = {"sparse": 0, "dense": 1, "unknown": 2}
RENDER_MODES = {"human": 1, "rgb_array": 2, "ansi": 4, "depth_array": 8}  # Bitmask values

def encode_metafeatures(env, check_stochasticity=False):
    """Extract and encode all meta-features as integers"""
    features = extract_metafeatures(env, check_stochasticity)
    encoded = []
    
    # 1. Observation space features (6 dimensions)
    encoded.append(OBS_SPACE_TYPES.get(features["observation_space_type"], 99))
    encoded.append(hash_shape(features["observation_space_shape"]))
    encoded.append(DTYPE_MAP.get(str(features["observation_space_dtype"]), 99))
    encoded.append(1 if features["is_image_observation"] else 0)
    encoded.append(features["state_space_size"])
    encoded.extend(encode_space_bounds(features["observation_space_bounds"]))

    # 2. Action space features (5 dimensions)
    encoded.append(ACTION_SPACE_TYPES.get(features["action_space_type"], 99))
    encoded.append(1 if features["is_action_discrete"] else 0)
    if features["is_action_discrete"]:
        encoded.append(features.get("num_actions", 0))
    else:
        encoded.append(hash_shape(features.get("action_space_shape", ())))
    encoded.extend(encode_space_bounds(features["action_space_bounds"]))

    # 3. Reward characteristics (3 dimensions)
    encoded.extend(encode_reward_range(features["reward_range"]))
    encoded.append(REWARD_DENSITY.get(features["reward_density"], 2))

    # 4. Episode properties (3 dimensions)
    encoded.append(1 if features["is_episodic"] else 0)
    encoded.append(features.get("max_episode_steps", 0))
    encoded.append(1 if features["is_goal_based"] else 0)

    # 5. Environment metadata (3 dimensions)
    encoded.append(encode_render_modes(features["available_render_modes"]))
    encoded.append(encode_stochasticity(features.get("is_deterministic")))
    encoded.append(1 if features["is_time_dependent"] else 0)

    return np.array(encoded, dtype=np.int32)

# Helper functions ------------------------------------------------------------

def hash_shape(shape):
    """Convert shape tuple to unique integer"""
    if not shape:
        return 0
    return int(hashlib.sha1(str(shape).encode()).hexdigest()[:8], 16) % 10**8

def encode_reward_range(reward_range):
    """Encode reward range into 2 integers (min, max) with special values:
    - -999999 for -inf
    - 999999 for inf
    - 0 for None/unknown
    """
    min_val, max_val = reward_range
    return [
        int(min_val * 1000) if not np.isinf(min_val) else -999999,
        int(max_val * 1000) if not np.isinf(max_val) else 999999
    ]

def encode_space_bounds(bounds):
    """Encode space bounds (low, high) into 2 integers"""
    low, high = bounds
    return [
        int(low * 1000) if low is not None and not np.isinf(low) else -999999,
        int(high * 1000) if high is not None and not np.isinf(high) else 999999
    ]

def encode_render_modes(modes):
    """Bitmask encoding of render modes"""
    return sum(RENDER_MODES.get(m, 0) for m in modes)

def encode_stochasticity(value):
    """Encode determinism status:
    0 - Non-deterministic
    1 - Deterministic 
    2 - Unknown
    """
    if value is None:
        return 2
    return 1 if value else 0

def extract_metafeatures(env, check_stochasticity=False):
    """
    Extract meta-features from a Gymnasium environment.
    
    Args:
        env: Gymnasium environment instance
        check_stochasticity: Whether to test for environment stochasticity (may take a few steps)
    
    Returns:
        Dictionary of meta-features
    """
    unwrapped_env = env.unwrapped
    meta_features = {}

    # Observation space features
    meta_features["observation_space_type"] = type(env.observation_space).__name__
    meta_features["observation_space_shape"] = env.observation_space.shape
    meta_features["observation_space_dtype"] = env.observation_space.dtype
    meta_features["is_image_observation"] = (
        isinstance(env.observation_space, spaces.Box) and 
        len(env.observation_space.shape) >= 2
    )
    meta_features["state_space_size"] = (
        env.observation_space.n if isinstance(env.observation_space, spaces.Discrete) else 0
    )
    meta_features["observation_space_bounds"] = (
        (env.observation_space.low[0], env.observation_space.high[0])
        if isinstance(env.observation_space, spaces.Box) and env.observation_space.low.size > 0
        else (None, None)
    )

    # Action space features
    meta_features["action_space_type"] = type(env.action_space).__name__
    meta_features["is_action_discrete"] = isinstance(env.action_space, spaces.Discrete)
    meta_features["num_actions"] = (
        env.action_space.n if meta_features["is_action_discrete"] else 0
    )
    meta_features["action_space_shape"] = (
        env.action_space.shape if isinstance(env.action_space, spaces.Box) else ()
    )
    meta_features["action_space_bounds"] = (
        (env.action_space.low[0], env.action_space.high[0])
        if isinstance(env.action_space, spaces.Box) and env.action_space.low.size > 0
        else (None, None)
    )

    # Reward characteristics
    try:
        meta_features["reward_range"] = getattr(unwrapped_env, "reward_range", (-float("inf"), float("inf")))
    except AttributeError:
        meta_features["reward_range"] = (-float("inf"), float("inf"))
    meta_features["reward_density"] = estimate_reward_density(unwrapped_env)

    # Episode properties
    try:
        max_steps = None
        if hasattr(unwrapped_env, "spec") and unwrapped_env.spec is not None:
            max_steps = getattr(unwrapped_env.spec, "max_episode_steps", None)
        meta_features["is_episodic"] = max_steps is not None
        meta_features["max_episode_steps"] = max_steps if max_steps is not None else 0
    except AttributeError:
        meta_features["is_episodic"] = False
        meta_features["max_episode_steps"] = 0
    meta_features["is_goal_based"] = check_goal_based(unwrapped_env)

    # Environment metadata
    try:
        meta_features["available_render_modes"] = getattr(unwrapped_env, "metadata", {}).get("render_modes", [])
    except AttributeError:
        meta_features["available_render_modes"] = []
    meta_features["is_deterministic"] = None  # Requires check_stochasticity logic if enabled
    meta_features["is_time_dependent"] = check_time_dependency(unwrapped_env)

    return meta_features

def estimate_reward_density(env):
    """
    Estimate reward density based on environment characteristics.
    Returns 'sparse', 'dense', or 'unknown'.
    """
    try:
        # Heuristic: Environments with large state spaces or specific names might indicate sparsity
        if hasattr(env, "spec") and env.spec is not None:
            env_name = env.spec.id.lower()
            if "maze" in env_name or "sparse" in env_name:
                return "sparse"
        # If reward range is very narrow or binary, assume dense
        reward_range = getattr(env, "reward_range", (-float("inf"), float("inf")))
        if reward_range[0] == reward_range[1] or abs(reward_range[1] - reward_range[0]) < 1.0:
            return "dense"
        return "unknown"
    except AttributeError:
        return "unknown"

def check_goal_based(env):
    """
    Check if the environment is goal-based (e.g., has a clear success condition).
    Returns True if goal-based, False otherwise.
    """
    try:
        env_name = env.spec.id.lower() if hasattr(env, "spec") and env.spec is not None else ""
        # Heuristic: Environments like maze, fetch, or goal in name are often goal-based
        return any(keyword in env_name for keyword in ["maze", "fetch", "goal", "reach"])
    except AttributeError:
        return False

def check_time_dependency(env):
    """
    Check if the environment has time-dependent dynamics.
    Returns True if time-dependent, False otherwise.
    """
    try:
        # Heuristic: Environments with time in metadata or specific names
        env_name = env.spec.id.lower() if hasattr(env, "spec") and env.spec is not None else ""
        return any(keyword in env_name for keyword in ["time", "season", "dynamic"])
    except AttributeError:
        return False