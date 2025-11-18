import matplotlib.pyplot as plt
import numpy as np

dqn_10_cm = np.array([[1.702, 0.0, 1.682, 1.76, 1.717, 1.363, 1.717, 1.717, 1.717, 1.717], [0.34, 0.0, 1.696, 1.531, 1.288, 1.558, 1.676, 1.558, 1.558, 1.549], [0.181, 0.0, 1.532, 0.368, 1.697, 1.712, 0.37, 1.697, 1.686, 1.697], [1.637, 0.0, 1.71, 1.587, 1.34, 1.587, 1.728, 1.062, 1.587, 1.587], [1.214, 1.705, 0.869, 1.216, 1.686, 1.707, 0.0, 1.71, 1.707, 1.707]])

dqn_10_sp = np.array([[0.782, 0.774, 0.776, 0.782, 0.778, 0.787, 0.778, 0.778, 0.778, 0.782], [0.769, 0.772, 0.76, 0.771, 0.781, 0.781, 0.785, 0.781, 0.781, 0.771], [0.767, 0.772, 0.764, 0.784, 0.776, 0.767, 0.776, 0.776, 0.789, 0.791], [0.786, 0.773, 0.775, 0.775, 0.782, 0.775, 0.77, 0.773, 0.775, 0.775], [0.771, 0.776, 0.77, 0.774, 0.792, 0.784, 0.774, 0.774, 0.784, 0.784]])

dqn_10_sm = np.array([[0.0, 0.0, 0.6, 0.2, 0.6, 1.0, 0.2, 0.2, 0.6, 0.2], [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.8, 1.0, 0.2, 1.0, 1.0, 0.2], [1.0, 0.0, 1.0, 0.4, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4], [0.8, 0.0, 0.8, 0.8, 1.0, 0.8, 0.0, 1.0, 0.8, 0.0]])

bo_10_sm = np.array([[0.4, 0.6, 0.2, 0.4, 0.8, 0.4, 0.2, 0.0, 0.0, 0.8], [0.6, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 1.0, 0.0], [1.0, 0.8, 1.0, 0.0, 0.8, 0.6, 0.0, 0.0, 0.8, 1.0], [0.6, 0.8, 0.2, 0.0, 0.0, 0.0, 1.0, 0.8, 0.0, 0.2], [0.2, 0.8, 1.0, 1.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0]])

bo_10_sp = np.array([[0.781, 0.768, 0.785, 0.776, 0.7849999999999999, 0.779, 0.776, 0.779, 0.7710000000000001, 0.78], [0.777, 0.772, 0.7849999999999999, 0.776, 0.784, 0.766, 0.795, 0.778, 0.779, 0.7689999999999999], [0.767, 0.773, 0.762, 0.771, 0.78, 0.779, 0.783, 0.774, 0.774, 0.779], [0.783, 0.795, 0.787, 0.7700000000000001, 0.787, 0.7709999999999999, 0.7869999999999999, 0.78, 0.7699999999999999, 0.792], [0.764, 0.7889999999999999, 0.788, 0.7729999999999999, 0.782, 0.7830000000000001, 0.764, 0.7759999999999999, 0.775, 0.7750000000000001]])

bo_10_cm = np.array([[0.0, 0.18, 0.538, 0.0, 1.0219999999999998, 0.5345, 0.0, 0.34750000000000003, 0.0, 0.0215], [0.176, 0.0, 0.861, 0.868, 0.6825, 0.8350000000000002, 0.5029999999999999, 0.0, 0.0, 0.341], [1.704, 1.0394999999999999, 0.3445, 0.0, 0.38649999999999995, 1.0219999999999998, 0.0, 0.014499999999999999, 1.7035, 1.6915], [0.337, 0.8684999999999998, 0.0, 0.0, 0.0, 0.0, 1.7570000000000001, 0.0, 0.0, 0.4995], [0.0, 0.8664999999999999, 1.1444999999999999, 1.7100000000000002, 0.7304999999999999, 1.3495, 0.178, 0.193, 0.0, 0.0]])

rs_10_sm = np.array([[0.4, 0.6, 0.2, 0.4, 0.8, 0.4, 0.2, 0.0, 0.0, 0.8], [0.6, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 1.0, 0.0], [1.0, 0.8, 1.0, 0.0, 0.8, 0.6, 0.0, 0.0, 0.8, 1.0], [0.6, 0.8, 0.2, 0.0, 0.0, 0.0, 1.0, 0.8, 0.0, 0.2], [0.2, 0.8, 1.0, 1.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0]])

rs_10_sp = np.array([[0.781, 0.768, 0.785, 0.776, 0.7849999999999999, 0.779, 0.776, 0.779, 0.7710000000000001, 0.78], [0.777, 0.772, 0.7849999999999999, 0.776, 0.784, 0.766, 0.795, 0.778, 0.779, 0.7689999999999999], [0.767, 0.773, 0.762, 0.771, 0.78, 0.779, 0.783, 0.774, 0.774, 0.779], [0.783, 0.795, 0.787, 0.7700000000000001, 0.787, 0.7709999999999999, 0.7869999999999999, 0.78, 0.7699999999999999, 0.792], [0.764, 0.7889999999999999, 0.788, 0.7729999999999999, 0.782, 0.7830000000000001, 0.764, 0.7759999999999999, 0.775, 0.7750000000000001]])

rs_10_cm = np.array([[0.0, 0.18, 0.538, 0.0, 1.0219999999999998, 0.5345, 0.0, 0.34750000000000003, 0.0, 0.0215], [0.176, 0.0, 0.861, 0.868, 0.6825, 0.8350000000000002, 0.5029999999999999, 0.0, 0.0, 0.341], [1.704, 1.0394999999999999, 0.3445, 0.0, 0.38649999999999995, 1.0219999999999998, 0.0, 0.014499999999999999, 1.7035, 1.6915], [0.337, 0.8684999999999998, 0.0, 0.0, 0.0, 0.0, 1.7570000000000001, 0.0, 0.0, 0.4995], [0.0, 0.8664999999999999, 1.1444999999999999, 1.7100000000000002, 0.7304999999999999, 1.3495, 0.178, 0.193, 0.0, 0.0]])

ppo_10_cm = np.array([[0.0, 1.548, 1.702, 0.539, 1.723, 1.524, 1.524, 1.702, 1.702, 1.702], [0.344, 1.531, 0.167, 1.724, 1.724, 1.03, 0.878, 1.685, 1.685, 1.685], [0.674, 1.054, 1.712, 1.626, 1.56, 0.368, 1.7, 1.532, 1.054, 1.712], [1.728, 1.704, 0.0, 0.684, 0.373, 0.684, 0.335, 1.706, 0.335, 0.713], [0.338, 1.678, 1.694, 1.71, 1.71, 1.012, 1.686, 1.686, 1.686, 1.686]])

ppo_10_sp = np.array([[0.772, 0.769, 0.773, 0.771, 0.774, 0.778, 0.789, 0.774, 0.778, 0.782], [0.768, 0.792, 0.772, 0.796, 0.79, 0.781, 0.779, 0.772, 0.796, 0.78], [0.78, 0.777, 0.767, 0.808, 0.771, 0.771, 0.791, 0.764, 0.784, 0.782], [0.77, 0.777, 0.773, 0.772, 0.773, 0.78, 0.777, 0.777, 0.78, 0.776], [0.784, 0.779, 0.774, 0.787, 0.786, 0.786, 0.786, 0.797, 0.778, 0.797]])

ppo_10_sm = np.array([[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.2, 1.0, 0.2, 1.0], [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.4, 0.6, 1.0, 0.0, 0.6, 1.0, 0.4, 0.4, 1.0, 0.4], [0.0, 1.0, 0.8, 0.6, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]])

def get_best_performance_run(array):
    maxes = np.max(array, axis=1)
    mean = np.mean(maxes) # Max performance per run
    std = np.std(maxes)
    return [mean, std]

def plot_anytime_performance(data_dict, algorithm_colors, shading_colors, confidence_level, title, y_label, x_label):
    first_key = next(iter(data_dict))
    
    num_trials = data_dict[first_key].shape[1]
    trials = np.arange(1, num_trials + 1)
    
    # --- 1. Calculate Statistics for each Algorithm ---
    stats = {}
    for name, data in data_dict.items():
        num_runs = data.shape[0]
        
        # 1. Calculate Cumulative Max for each run (Best reward so far)
        cumulative_max_runs = np.maximum.accumulate(data, axis=1)

        # 2. Calculate the Mean across all runs
        mean_performance = np.mean(cumulative_max_runs, axis=0)

        # 3. Calculate the variability (Standard Deviation or Standard Error)
        std_performance = np.std(cumulative_max_runs, axis=0)
        
        if confidence_level == 'stderr' and num_runs > 1:
            # Standard Error
            variability = std_performance / np.sqrt(num_runs)
        else:
            # Standard Deviation (default)
            variability = std_performance
        
        stats[name] = {
            'mean': mean_performance,
            'lower': mean_performance - variability,
            'upper': mean_performance + variability,
            'line_color': algorithm_colors.get(name, 'gray'),
            'shading_color': shading_colors.get(name, 'lightgray') # Use defined shading color
        }

    # --- 2. Plotting ---

    plt.figure(figsize=(9, 6))

    # Plot the shaded area first (Confidence Interval)
    for name, s in stats.items():
        plt.fill_between(
            trials, 
            s['lower'], 
            s['upper'], 
            color=s['shading_color'], # Using the new shading_color
            alpha=0.2, 
            # IMPORTANT: Removed 'label' to prevent shaded region from appearing in the legend
            linewidth=0.0
        )

    # Plot the mean line second (On top of the shading)
    # This plot call includes the label for the line only, ensuring a clean legend.
    for name, s in stats.items():
        plt.plot(
            trials, 
            s['mean'], 
            color=s['line_color'], 
            label=name, # Only the line gets the label
            linewidth=2
        )

    # --- 3. Styling and Labels ---

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)

    plt.xticks(trials) 
    plt.xlim(1, num_trials)

    # Legend in the top-left, showing only the line labels
    plt.legend(loc='upper left', frameon=False) 

    # Clean up figure appearance (remove top/right borders)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add subtle horizontal grid lines (like the original image)
    plt.grid(axis='y', linestyle='dotted', alpha=0.5, linewidth=0.5)

    plt.tight_layout()
    plt.show()

environments = ["ComplexMaze-v0", "Sepsis/ICU-Sepsis-v2", "SimpleMaze-v0"]
methods = ["Random search", "Bayesian optimisation", "DQN", "PPO"]

means = np.array([
    [get_best_performance_run(rs_10_cm)[0], get_best_performance_run(bo_10_cm)[0], get_best_performance_run(dqn_10_cm)[0], get_best_performance_run(ppo_10_cm)[0]],  # ComplexMaze-v0
    [get_best_performance_run(rs_10_sp)[0], get_best_performance_run(bo_10_sp)[0], get_best_performance_run(dqn_10_sp)[0], get_best_performance_run(ppo_10_sp)[0]],# Sepsis/ICU-Sepsis-v2
    [get_best_performance_run(rs_10_sm)[0], get_best_performance_run(bo_10_sm)[0], get_best_performance_run(dqn_10_sm)[0], get_best_performance_run(ppo_10_sm)[0]]   # SimpleMaze-v0
])

# Standard deviations for each environment × method
stds = np.array([
    [get_best_performance_run(rs_10_cm)[1], get_best_performance_run(bo_10_cm)[1], get_best_performance_run(dqn_10_cm)[1], get_best_performance_run(ppo_10_cm)[1]],
    [get_best_performance_run(rs_10_sp)[1], get_best_performance_run(bo_10_sp)[1], get_best_performance_run(dqn_10_sp)[1], get_best_performance_run(ppo_10_sp)[1]],
    [get_best_performance_run(rs_10_sm)[1], get_best_performance_run(bo_10_sm)[1], get_best_performance_run(dqn_10_sm)[1], get_best_performance_run(ppo_10_sm)[1]]
])

n_runs = 5
ci95 = stds

print(means)
print(stds)

# ---------------------------
# Plot one figure per environment
# ---------------------------
for i, env in enumerate(environments):
    plt.figure(figsize=(10, 7))
    
    x = np.arange(len(methods))
    plt.bar(x, means[i], yerr=ci95[i], capsize=5, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(methods))))
    
    # Annotate mean values
    for j, val in enumerate(means[i]):
        plt.text(j, val + ci95[i][j] + 0.02 * max(means[i]), f"{val:.2f}", 
                ha="center", va="bottom", fontsize=10)
    
    plt.xticks(x, methods)
    plt.ylabel("Return")
    plt.title(f"{env}: Mean of each run's best return (10 trials)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()

############################################################################################

# data_to_plot = {
#     'Random search': rs_10_sm,
#     'Bayesian optimisation': bo_10_sm,
#     'DQN': dqn_10_sm,
#     'PPO': ppo_10_sm,
# }

# # 2. Define Colors for Plotting
# colors = {
#     'Random search': 'orange',
#     'Bayesian optimisation': 'black',
#     'DQN': 'green',
#     'PPO': 'blue',
# }

# shading_colors = {
#     'Random search': 'peachpuff',
#     'Bayesian optimisation': 'silver',
#     'DQN': 'palegreen',
#     'PPO': 'lightskyblue',
# }

# # 3. Generate the Plot!
# plot_anytime_performance(
#     data_to_plot,
#     colors,
#     confidence_level='std', # Use 'std' for wider bands, 'stderr' for narrower
#     title='Performance trajectory of each method across 5 runs of 10 trials (SimpleMaze-v0)',
#     y_label='Best Cumulative Reward (Mean $\\pm$ SD)',
#     x_label="Trial number",
#     shading_colors=shading_colors
# )
