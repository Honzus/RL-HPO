import optuna
from agent import TabularRL
import numpy as np
import gymnasium as gym
import icu_sepsis
import simplemaze
import complexmaze
import random

SEEDS = [0, 1, 2, 3, 4]

def objective(trial, s):
    epsilon = trial.suggest_float('epsilon', 0, 1)
    learning_rate = trial.suggest_float('learning_rate', 0, 1)
    discount_factor = 0.99

    all_final_rewards = []

    for seed in SEEDS:
        np.random.seed(s*100 + seed)
        random.seed(s*100 + seed)
        env = gym.make('SimpleMaze-v0') # Sepsis/ICU-Sepsis-v2, ComplexMaze-v0
        _, _ = env.reset(seed=s*100 + seed)
        env.action_space.seed(s*100 + seed)
        env.observation_space.seed(s*100 + seed)

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

all_run_results = []
num_runs = 5

for run in range(num_runs):
    print(f"\n=== Starting Run {run + 1}/{num_runs} ===")
    
    # Set seed for this run
    run_seed = run
    np.random.seed(run_seed)
    
    # Create study for this run
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100)
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.GPSampler(seed=run_seed))
    
    # Optimize
    study.optimize(lambda trial: objective(trial, s=run_seed), n_trials=10)
    
    # Store results
    run_results = {
        'run_seed': run_seed,
        'best_trial': study.best_trial,
        'best_value': study.best_trial.value,
        'best_params': study.best_trial.params
    }
    all_run_results.append(run_results)
    
    print(f"Run {run + 1} completed:")
    print(f"  Best value: {study.best_trial.value:.4f}")
    print(f"  Best params: {study.best_trial.params}")

# === AGGREGATE RESULTS ACROSS ALL RUNS ===
print(f"\n=== AGGREGATED RESULTS OVER {num_runs} RUNS ===")

# Extract best values from all runs
best_values = [run['best_value'] for run in all_run_results]
avg_best_value = np.mean(best_values)
std_best_value = np.std(best_values)
min_best_value = np.min(best_values)
max_best_value = np.max(best_values)

print(f"Average best value: {avg_best_value:.4f} ± {std_best_value:.4f}")
print(f"Value range: [{min_best_value:.4f}, {max_best_value:.4f}]")

# Find most common best configuration
param_counts = {}
for run in all_run_results:
    params_tuple = tuple(sorted(run['best_params'].items()))
    param_counts[params_tuple] = param_counts.get(params_tuple, 0) + 1

if param_counts:
    most_common_params = max(param_counts, key=param_counts.get)
    most_common_count = param_counts[most_common_params]
    print(f"Most common best config: {dict(most_common_params)} (appeared in {most_common_count}/{num_runs} runs)")

# Detailed run-by-run results
print(f"\nDetailed results:")
for i, run in enumerate(all_run_results):
    print(f"  Run {i+1}: {run['best_value']:.4f} (lr={run['best_params']['learning_rate']:.4f}, eps={run['best_params']['epsilon']:.4f})")
