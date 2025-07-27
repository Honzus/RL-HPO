import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import argparse
import os
import gymnasium as gym
from common.cmd_util import make_meta_env
from agent.ppo.build_graph import lstm_to_mlp, build_policy
from agent.deepq.utils import ObservationInput
from agent.deepq.simple import ActWrapper
import common.tf_util as U
import logger
import copy
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
        action = act_wrapper(obs_batch, [current_seq_len])[0][0]
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

    path = "checkpoints2"
    logger.configure(path)

    # Create the environment (same as used in training)
    training_env = make_meta_env(args.env_id, seed=args.seed)
    N_t = 4  # Number of hyperparameter dimensions
    N_f = 21  # Number of meta-features
    max_seq_len = training_env.observation_space.shape[0]

    sess = tf.Session()
    sess.__enter__()

    # Define the model architecture (must match training)
    actor_network, value_network = lstm_to_mlp(
        cell=(32, N_t, N_f),  # Make sure these parameters match your training
        aktiv=tf.nn.tanh,
        hiddens=[128],  # Make sure this matches your training
        max_length=max_seq_len,
    )

    # Create the observation placeholder function
    def make_obs_ph(name):
        return ObservationInput(training_env.observation_space, name=name)

    # Build the policy networks
    act, train, update_old_policy, compute_values = build_policy(
        make_obs_ph=make_obs_ph,
        actor_network=actor_network,
        value_network=value_network,
        num_actions=training_env.action_space.n,
    )

    # Store act parameters for saving/loading
    act_params = {
        'make_obs_ph': make_obs_ph,
        'actor_network': actor_network,
        'value_network': value_network,
        'num_actions': training_env.action_space.n,
        'N_t': N_t
    }

    # Create the act wrapper
    trained_act_wrapper = ActWrapper(act, act_params)

    # Initialize the model
    U.initialize()
    update_old_policy()

    model_dir = "../Users2/janrichtr/Desktop/RL-HPO/batch_size-64/buffer_size-2048/cell-32/checkpoint_freq-10000/clip_ratio-0.2/env_id-nnMeta-v40/gamma-0.99/lam-0.95/log_interval-1/lr-0.0002/nhidden-128/num_timesteps-1000000/seed-0/train_epochs-10/train_iters-4"

    # Load the trained model
    model_file = os.path.join(model_dir, "model")
    U.load_state(model_file)
    logger.log('Loaded model from {}'.format(model_file))

    # Optimize hyperparameters for the new environment
    best_config, best_reward = optimize_hyperparameters(
        env_name=args.new_env_id,
        training_env=training_env,
        act_wrapper=trained_act_wrapper,
        max_seq_len=max_seq_len,
        meta_dim=N_f,
        N_t=N_t,
        n_t_dim=N_t + N_f,
        num_actions=training_env.action_space.n,
        max_trials=10
    )

    print("\nFinal Results:")
    print(f"Best configuration: {best_config}")
    print(f"Best reward: {best_reward:.4f}")

if __name__ == '__main__':
    main()