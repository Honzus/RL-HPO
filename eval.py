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
import kellycoinflip
import nchain
from TabularAgents import TabularRL
import utils

SEEDS = [0, 1, 2, 3, 4]

def prepare_initial_observation(max_seq_len, n_t_dim, meta_dim, meta_features):
    """Initialize the observation matrix with NaNs and meta-features."""
    obs = np.zeros((max_seq_len, n_t_dim), dtype=np.float32)  # ← Use zeros instead of NaN
    obs[:, -meta_dim:] = meta_features
    return obs

def eval(env_name, config):
    learning_rate, discount_factor, epsilon = config

    all_final_rewards = []

    for seed in SEEDS:
        np.random.seed(seed)
        env = gym.make(env_name) # NChain-v0, KellyCoinflip-v0, Sepsis/ICU-Sepsis-v2

        agent = TabularRL(env, learning_rate, epsilon, discount_factor)

        episode_rewards = []
        window_size = 5000        
        # Track convergence for early stopping
        convergence_window = 2000
        convergence_threshold = 0.001
        
        for episode in range(10001):
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

            # Early stopping
            if len(episode_rewards) >= window_size:
                avg_reward = np.mean(episode_rewards[-window_size:])
                
                # Remove for 8x8
                if len(episode_rewards) >= convergence_window:
                    recent_avg = np.mean(episode_rewards[-convergence_window//2:])
                    older_avg = np.mean(episode_rewards[-convergence_window:-convergence_window//2])
                    if abs(recent_avg - older_avg) < convergence_threshold:
                        break

        # Evaluation
        agent.epsilon = 0
        eval_rewards = []
        
        num_eval_episodes = 200 
        
        eval_rng = np.random.RandomState(seed * 10000)
        
        for eval_episode in range(num_eval_episodes):
            eval_seed = eval_rng.randint(0, 1000000)
            
            obs, info = env.reset(seed=eval_seed)
            done = False
            ep_reward = 0
            
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
                obs = next_obs
                
            eval_rewards.append(ep_reward)

        all_final_rewards.append(np.mean(eval_rewards))
        
        env.close()

    mean_reward = np.mean(all_final_rewards)
    return mean_reward

# Main optimization loop
def optimize_hyperparameters(env_name, training_env, act_wrapper, max_seq_len, meta_dim, N_t, n_t_dim,
                             num_actions, max_trials):
    """Uses the Hyp-RL agent to optimize RL agent hyperparameters in a single episode."""
    
    hpo_history = []
    best_reward = -float('inf')
    best_config = None
    best_action = None

    # Get action meanings from the training environment
    training_action_meanings = training_env.env.get_action_meanings2()

    # === CRITICAL FIX: Use meta-environment's observation directly ===
    # Reset the meta-environment for a new episode
    training_env.reset()
    
    # Initialize observation exactly like DQN training
    obs = np.zeros(shape=training_env.observation_space.shape)
    available_datasets = list(training_env.ale.metadata.keys())
    dataset_idx = available_datasets[0] if available_datasets else 0
    
    # Set first row exactly like DQN training: [NaN, NaN, ..., meta_features]
    obs[0, :] = np.append(np.repeat(np.NaN, repeats=N_t), training_env.ale.metadata[dataset_idx]['features']).reshape(1, -1)
    obs = np.nan_to_num(obs, copy=True, nan=0.0)  # Replace NaN with 0
    
    # Run the episode (each trial is one step)
    for step in range(max_trials):
        print(f"\n=== Trial {step + 1}/{max_trials} ===")
        
        # Get current sequence length
        current_seq_len = training_env.env._get_ep_len()
        
        # Prepare observation batch for act function
        obs_batch = obs[None, :, :]  # Shape: (1, max_sequence_length, hyper_parameter_size)
        
        # Get action from meta-agent (this is one step in the episode)
        action = act_wrapper(obs_batch, [current_seq_len], stochastic=False, update_eps=0)[0]

        # Validate action
        if not (0 <= action < num_actions):
            print(f"  Error: Action {action} out of bounds. Skipping trial.")
            continue

        # Get the hyperparameters from the TRAINING environment's action space
        config = training_action_meanings[action]
        print(f"  Selected action: {action}")
        print(f"  Config: {config}")

        # Evaluate on target environment
        reward = eval(env_name, config)
        
        # Store the result
        hpo_history.append((config, reward))
        print(f"  Reward: {reward}")

        # Update best configuration
        if reward > best_reward:
            best_reward = reward
            best_config = config
            best_action = action
            print(f"  *** NEW BEST! ***")

        print(f"  Best config so far: {best_config}")
        print(f"  Best performance so far: {best_reward}")

        # === CRITICAL FIX: Step the meta-environment exactly like DQN training ===
        # This updates the observation matrix automatically
        new_state, meta_reward, done, info = training_env.step(action, dataset_idx)
        
        # Update observation exactly like DQN training
        new_obs = np.copy(obs)
        new_obs[training_env.env._get_ep_len(), :] = np.append(new_state, np.append(meta_reward, training_env.ale.metadata[dataset_idx]['features'])).reshape(1, -1)
        obs = new_obs
        
        # Check episode termination
        if done:
            print(f"  Meta-episode terminated after {step + 1} trials")
            break

    # Find the best configuration
    if hpo_history:
        best_config, best_performance = max(hpo_history, key=lambda x: x[1])
        print(f"\n=== FINAL RESULTS ===")
        print(f"Best configuration: {best_config}")
        print(f"Best performance: {best_performance:.3f}")
        print(f"Total trials completed: {len(hpo_history)}")
    else:
        best_config, best_performance = None, -float('inf')
        print(f"\nNo trials completed!")

    return best_config, best_performance

def main():
    # Parse arguments (if needed for environment or other settings)
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', help='environment ID', default='nnMeta-v40')
    parser.add_argument('--new_env_id', help='environment ID', default='Sepsis/ICU-Sepsis-v2') # Sepsis/ICU-Sepsis-v2
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
    checkpoint_dir = "Users/janrichtr/Desktop/RL-HPO/buffer_size-2000/cell-32/checkpoint_freq-10000/dueling-0/ei-True/env_id-nnMeta-v40/learning_starts-2000/lr-0.0001/new_env_id-Taxi-v3/nhidden-128/num_timesteps-2000000/prioritized-1/prioritized_replay_alpha-0.6/seed-0/target_network_update_freq-500/train_freq-1"

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