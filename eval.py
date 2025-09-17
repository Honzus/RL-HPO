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
import complexmaze
import simplemaze
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

    # Get target environment meta-features
    target_env = gym.make(env_name)
    target_meta_features = utils.encode_metafeatures(target_env)
    target_env.close()

    # Reset the meta-environment for a new episode
    training_env.reset()

    # Initialize observation matrix
    obs = np.zeros(shape=training_env.observation_space.shape)

    # Set initial state with target meta-features
    obs[0, :] = np.append(np.repeat(np.NaN, repeats=N_t), target_meta_features).reshape(1, -1)
    obs = np.nan_to_num(obs, copy=True, nan=0.0)

    start_eps = 0.7
    end_eps = 0.1

    # Run the episode (each trial is one step)
    for step in range(max_trials):
        print(f"  Trial {step + 1}/{max_trials}")
        
        # Get current sequence length
        current_seq_len = training_env.env._get_ep_len()
        
        # Prepare observation batch for act function
        obs_batch = obs[None, :, :]
        
        current_eps = start_eps - (start_eps - end_eps) * (step / max_trials)

        # Get action from meta-agent
        action = act_wrapper(obs_batch, [current_seq_len], stochastic=True, update_eps=current_eps)[0]

        # Validate action
        if not (0 <= action < num_actions):
            print(f"    Error: Action {action} out of bounds. Skipping trial.")
            continue

        # Get the hyperparameters
        config = training_action_meanings[action]
        print(f"    Selected action: {action}")
        print(f"    Config: {config}")

        # Evaluate on target environment
        reward = eval(env_name, config)
        
        # Store the result
        hpo_history.append((config, reward))
        print(f"    Reward: {reward:.3f}")

        # Update best configuration
        if reward > best_reward:
            best_reward = reward
            best_config = config
            best_action = action
            print(f"    *** NEW BEST! ***")

        print(f"    Best config so far: {best_config}")
        print(f"    Best performance so far: {best_reward}")

        # Step the meta-environment to get the state representation
        available_datasets = list(training_env.ale.metadata.keys())
        dataset_idx = available_datasets[0] if available_datasets else 0
        new_state, meta_reward, done, info = training_env.step(action, dataset_idx)
        
        # Update the observation matrix with the complete trial information
        current_row = training_env.env._get_ep_len()
        if current_row < obs.shape[0]:
            # Format: [config, reward, meta_features]
            trial_data = np.append(config, reward)
            trial_data = np.append(trial_data, target_meta_features)
            
            # Ensure we don't exceed the observation space
            if len(trial_data) <= obs.shape[1]:
                obs[current_row, :len(trial_data)] = trial_data
            else:
                # Truncate if necessary
                obs[current_row, :] = trial_data[:obs.shape[1]]

        # Override episode termination for HPO
        if step + 1 >= max_trials:
            print(f"    HPO completed after {step + 1} trials")
            break
        
        if done:
            print(f"    Meta-environment terminated early, but continuing HPO...")

    # === FINAL RESULTS ===
    print(f"\n=== FINAL RESULTS ===")

    if hpo_history:
        best_config, best_performance = max(hpo_history, key=lambda x: x[1])
        print(f"Best configuration: {best_config}")
        print(f"Best performance: {best_performance:.3f}")
        print(f"Total trials completed: {len(hpo_history)}")
        
        # Return the best config and performance
        return best_config, best_performance
        
    else:
        print("No trials completed!")
        return None, -float('inf')

def main():
    # Parse arguments (if needed for environment or other settings)
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', help='environment ID', default='nnMeta-v40')
    parser.add_argument('--new_env_id', help='environment ID', default='ComplexMaze-v0') # Sepsis/ICU-Sepsis-v2
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()

    path = "checkpoints"  # Adjust to your desired log directory
    logger.configure(path)
        
    # Create the environment with this run's seed
    training_env = make_meta_env(args.env_id, seed=args.seed)
    N_t = 4
    N_f = 21

    sess = tf.Session()
    sess.__enter__()

    # Define the model architecture (must match training)
    model = deepq.models.lstm_to_mlp(
        cell=(32, N_t, N_f),
        aktiv=tf.nn.tanh,
        hiddens=[128],
        max_length=training_env.observation_space.shape[0],
        dueling=False,
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

    checkpoint_dir = "Users/janrichtr/Desktop/RL-HPO/buffer_size-2000/cell-32/checkpoint_freq-10000/dueling-0/ei-True/env_id-nnMeta-v40/learning_starts-2000/lr-0.0001/new_env_id-Taxi-v3/nhidden-128/num_timesteps-2000000/prioritized-1/prioritized_replay_alpha-0.6/seed-0/target_network_update_freq-500/train_freq-1"

    U.load_state(os.path.join(checkpoint_dir, "model"))

    max_length = training_env.observation_space.shape[0]

    # Run HPO for this seed
    best_config, best_performance = optimize_hyperparameters(
        env_name=args.new_env_id,
        training_env=training_env,
        act_wrapper=trained_act_wrapper,
        max_seq_len=max_length,
        meta_dim=N_f,
        N_t=N_t,
        n_t_dim=N_t + N_f,
        num_actions=training_env.action_space.n,
        max_trials=50
    )
    
    # Clean up session for next run
    sess.close()

if __name__ == '__main__':
    main()