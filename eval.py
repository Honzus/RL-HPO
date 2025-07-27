import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from agent import deepq
import gymnasium as gym
from common.misc_util import set_global_seeds
import argparse
import logger
import numpy as np
import os
import common.tf_util as U
from agent.deepq.utils import ObservationInput
from common.cmd_util import make_meta_env
from agent.deepq.simple import ActWrapper
import icu_sepsis
import minigrid
from TabularAgents import TabularRL
import utils

def prepare_initial_observation(max_seq_len, n_t_dim, meta_dim, meta_features):
    """Initialize the observation matrix with NaNs and meta-features."""
    obs = np.full((max_seq_len, n_t_dim), np.nan, dtype=np.float32)
    obs[:, -meta_dim:] = meta_features
    return obs

def eval(env, config):
    seed = 0
    np.random.seed(seed)
    learning_rate, discount_factor, epsilon = config

    agent = TabularRL(env, learning_rate, epsilon, discount_factor)

    episode_rewards = []
    for episode in range(20001):
        obs, info = env.reset(seed=seed + episode)
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            action2 = agent.select_action(next_obs)
            agent.update_sarsa(obs, action, next_obs, action2, reward, terminated)
            total_reward += reward
            done = terminated or truncated
            obs = next_obs
        episode_rewards.append(total_reward)

    agent.epsilon = 0  # Greedy policy
    rewards = []
    for run in range(10):
        total_rewards = []
        for episode in range(100):
            obs, info = env.reset(seed=run * 100 + episode)
            done = False
            episode_reward = 0
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                obs = next_obs
            total_rewards.append(episode_reward)
        rewards.append(np.mean(total_rewards))

    return np.mean(rewards)

def eval2(env, config):
    seed = 0
    np.random.seed(seed)
    learning_rate, discount_factor, epsilon = config

    agent = TabularRL(env, learning_rate, epsilon, discount_factor)

    episode_rewards = []
    for episode in range(2001):
        obs, info = env.reset(seed=seed + episode)
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            action2 = agent.select_action(next_obs)
            agent.update_sarsa(obs, action, next_obs, action2, reward, terminated)
            total_reward += reward
            done = terminated or truncated
            obs = next_obs
        episode_rewards.append(total_reward)

    agent.epsilon = 0  # Greedy policy
    rewards = []
    for run in range(10):
        total_rewards = []
        for episode in range(100):
            obs, info = env.reset(seed=run * 100 + episode)
            done = False
            episode_reward = 0
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                obs = next_obs
            total_rewards.append(episode_reward)
        rewards.append(np.mean(total_rewards))

    return np.mean(rewards)

# Main optimization loop
def optimize_hyperparameters(env_name, training_env, act_wrapper, max_seq_len, meta_dim, N_t, n_t_dim,
                             num_actions, max_trials):
    """Uses the Hyp-RL agent to optimize RL agent hyperparameters.

    Args:
        env_name: Name of the new environment to optimize
        training_env: The environment instance used during training (for action meanings)
        act_wrapper: The trained act function
        ...
    """
    # Create the new environment for optimization
    env = gym.make(env_name)
    meta_features = utils.encode_metafeatures(env)
    current_obs_matrix = prepare_initial_observation(max_seq_len, n_t_dim, meta_dim, meta_features)
    hpo_history = []
    best_reward = -float('inf')
    best_config = None

    # Get action meanings from the training environment
    training_action_meanings = training_env.env.get_action_meanings2()

    # Initialize new environment
    obs, info = env.reset()

    # Run trials
    for i in range(max_trials):
        # Get current state
        print(f"\nIteration {i + 1}/{max_trials}")
        # Determine current sequence length
        current_seq_len = max(1, min(len(hpo_history), max_seq_len))
        # Prepare observation batch for act function
        obs_batch = current_obs_matrix[None, :, :]  # Shape: (1, max_seq_len, n_t_dim)
        # Get action (hyperparameter config index) from Hyp-RL agent
        action = act_wrapper(obs_batch, [current_seq_len], stochastic=False, update_eps=0)[0]
        print(f"\nSelected action: {action}")

        # Validate action
        if not (0 <= action < num_actions):
            print(f"  Error: Action {action} out of bounds. Skipping iteration.")
            continue

        # Get the hyperparameters from the TRAINING environment's action space
        config = training_action_meanings[action]
        print(f"  Action {action} corresponds to hyperparameters (from training env):")
        print(f"  {config}")

        # Evaluate the configuration on the NEW environment
        reward = eval2(env, config)
        hpo_history.append((config, reward))

        # Update best configuration
        if reward > best_reward:
            best_reward = reward
            best_config = config
            best_action = action

        # Update observation matrix
        if len(hpo_history) <= max_seq_len:
            idx = len(hpo_history) - 1
            # Store hyperparameters in first 3 positions
            current_obs_matrix[idx, :N_t - 1] = config  # N_t-1 because last position is for reward
            # Store reward in the 4th position
            current_obs_matrix[idx, N_t - 1] = reward
            # Meta-features already set during initialization
        else:
            # Shift matrix to keep the last max_seq_len trials
            current_obs_matrix[:-1, :] = current_obs_matrix[1:, :]
            current_obs_matrix[-1, :N_t - 1] = config
            current_obs_matrix[-1, N_t - 1] = reward

        print(f"  Reward: {reward:.2f}")

    # Find the best configuration
    best_config, best_performance = max(hpo_history, key=lambda x: x[1])
    print("\nFinal Results:")
    print(f"Best Configuration: {best_config}")
    print(f"Best Performance: {best_performance:.2f}")

    env.close()
    return best_config, best_performance

def main():
    # Parse arguments (if needed for environment or other settings)
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', help='environment ID', default='nnMeta-v40')
    parser.add_argument('--new_env_id', help='environment ID', default='MiniGrid-DoorKey-5x5-v0') #MiniGrid-DoorKey-5x5-v0, MiniGrid-Empty-5x5-v0, Sepsis/ICU-Sepsis-v2
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()

    path = "checkpoints"  # Adjust to your desired log directory
    logger.configure(path)

    # Create the environment (same as used in training)
    training_env = make_meta_env(args.env_id, seed=args.seed)
    N_t = 4
    N_f = 21

    sess = tf.Session()
    sess.__enter__()

    # Define the model architecture (must match training)
    model = deepq.models.lstm_to_mlp(
        cell=(32, N_t, N_f),  # Make sure these parameters match your training
        aktiv=tf.nn.tanh,
        hiddens=[128],  # Make sure this matches your training
        max_length=training_env.observation_space.shape[0],
        dueling=False,  # Make sure this matches your training
    )

    # Create the observation placeholder function
    def make_obs_ph(name):
        return ObservationInput(training_env.observation_space, name=name)

    # Build the training functions
    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=model,
        num_actions=training_env.action_space.n,
        optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.95, epsilon=0.01),
        gamma=0.99,
        grad_norm_clipping=10,
        param_noise=False,
        session=sess
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': model,
        'num_actions': training_env.action_space.n,
    }

    # Create the act wrapper
    trained_act_wrapper = ActWrapper(act, act_params)

    # Initialize the model
    U.initialize()
    update_target()

    # Initialize the model
    U.initialize()
    update_target()

    checkpoint_dir = "../Users/janrichtr/Desktop/RL-HPO/buffer_size-100/cell-32/checkpoint_freq-10000/dueling-0/ei-True/env_id-nnMeta-v40/learning_starts-100/lr-0.001/new_env_id-Taxi-v3/nhidden-128/num_timesteps-100000/prioritized-0/prioritized_replay_alpha-0.6/seed-0/target_network_update_freq-10/train_freq-1"

    U.load_state(os.path.join(checkpoint_dir, "model"))

    max_length = training_env.observation_space.shape[0]

    best_config, best_performance = optimize_hyperparameters(
        env_name=args.new_env_id,  # New environment to optimize
        training_env=training_env,  # Pass the training environment for action meanings
        act_wrapper=trained_act_wrapper,
        max_seq_len=max_length,
        meta_dim=N_f,
        N_t=N_t,
        n_t_dim=N_t + N_f,
        num_actions=training_env.action_space.n,
        max_trials=10
    )

if __name__ == '__main__':
    main()