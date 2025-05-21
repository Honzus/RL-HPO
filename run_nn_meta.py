#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:19:57 2019

@author: hsjomaa
"""

from agent import deepq
import gymnasium as gym
from common.misc_util import set_global_seeds
import argparse
import logger
import numpy as np
from common.cmd_util import  make_meta_env
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from common.utils import model_dir
from TabularAgents import TabularRL

def prepare_initial_observation(max_seq_len, n_t_dim, meta_dim, meta_features):
    """Initialize the observation matrix with NaNs and meta-features."""
    obs = np.full((max_seq_len, n_t_dim), np.nan, dtype=np.float32)
    obs[:, -meta_dim:] = meta_features
    return obs

def train_and_evaluate(env, config):
    seed = 0
    np.random.seed(seed)
    """Trains a DQN agent with given hyperparameters and returns its performance."""
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

# Main optimization loop
def optimize_hyperparameters(env_name, training_env, act_wrapper, max_seq_len, meta_dim, meta_features, N_t, n_t_dim, num_actions, max_trials):
    """Uses the Hyp-RL agent to optimize RL agent hyperparameters.
    
    Args:
        env_name: Name of the new environment to optimize
        training_env: The environment instance used during training (for action meanings)
        act_wrapper: The trained act function
        ...
    """
    # Create the new environment for optimization
    env = gym.make(env_name)
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
        reward = train_and_evaluate(env, config)
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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='nnMeta-v40')
    parser.add_argument('--new_env_id', help='environment ID for optimization', default='Taxi-v3')  # New environment to optimize
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=0)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--dueling', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e1)) #10e7
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--learning_starts', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--target_network_update_freq', type=int, default=10)
    parser.add_argument('--cell', type=int, default=32)
    parser.add_argument('--nhidden', type=int, default=128)
    parser.add_argument('--ei', default='True')
    
    args = parser.parse_args()
    args.buffer_size = args.learning_starts
    if args.env_id.startswith('nn'):
        N_t = 4; N_f = 21
    elif args.env_id.startswith('svm'):
        N_t = 7; N_f = 3
    checkpoint_path = model_dir(args)
    logger.configure(checkpoint_path)
    set_global_seeds(args.seed)
    
    # Create training environment
    training_env = make_meta_env(args.env_id, seed=args.seed)
    max_length = training_env.observation_space.shape[0]

    model = deepq.models.lstm_to_mlp(
        cell=(args.cell,N_t,N_f),
        aktiv = tf.nn.tanh,
        hiddens=[args.nhidden],
        max_length = max_length,
        dueling=bool(args.dueling),
    )

    trained_act_wrapper = deepq.learn(
        training_env,
        q_func=model,
        lr=args.lr,
        max_timesteps=args.num_timesteps,
        buffer_size=args.buffer_size,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=args.train_freq,
        learning_starts=args.learning_starts,
        target_network_update_freq=args.target_network_update_freq,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        prioritized_replay_alpha=args.prioritized_replay_alpha,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_path=checkpoint_path.replace('/logs/','/checkpoints/'),
        ei=eval(args.ei),
        N_t=N_t
    )

    # Now optimize hyperparameters on new environment while using training env for action meanings
    best_config, best_performance = optimize_hyperparameters(
        env_name=args.new_env_id,  # New environment to optimize
        training_env=training_env,  # Pass the training environment for action meanings
        act_wrapper=trained_act_wrapper,
        max_seq_len=max_length,
        meta_dim=N_f,
        meta_features=[1,0,3,0,500,-999999,999999,0,1,6,-999999,999999,-999999,999999,2,0,0,0,7,2,0],  # You'll need to define this
        N_t=N_t,
        n_t_dim= N_t + N_f,
        num_actions=training_env.action_space.n,
        max_trials=5
    )

    print("\nOptimization Results:")
    print(f"Best Config: {best_config}")
    print(f"Best Performance: {best_performance}")

    training_env.close()

if __name__ == '__main__':
    main()

