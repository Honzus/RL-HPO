import matplotlib.pyplot as plt
import numpy as np

dqn_10_cm = np.array([
        [1.402, 0.000, 1.744, 1.674, 1.403, 1.722, 1.552, 1.713, 1.683, 1.541],
        [0.342, 0.000, 1.706, 1.687, 1.057, 1.695, 1.734, 1.353, 1.682, 1.391],
        [0.334, 0.000, 1.713, 0.504, 1.200, 1.736, 1.736, 1.739, 1.731, 1.698],
        [1.727, 0.000, 1.695, 1.662, 1.232, 1.679, 1.718, 1.169, 1.419, 1.565],
        [1.219, 1.347, 0.565, 1.183, 1.711, 1.359, 0.000, 1.736, 1.527, 1.537]
    ])

dqn_10_sp = np.array([[0.789, 0.774, 0.78, 0.773, 0.786, 0.763, 0.791, 0.787, 0.775, 0.77], [0.776, 0.767, 0.776, 0.785, 0.777, 0.782, 0.777, 0.777, 0.788, 0.782], [0.773, 0.784, 0.771, 0.774, 0.775, 0.796, 0.776, 0.78, 0.797, 0.78], [0.783, 0.78, 0.778, 0.777, 0.777, 0.782, 0.772, 0.789, 0.797, 0.778], [0.775, 0.782, 0.799, 0.777, 0.787, 0.782, 0.772, 0.798, 0.79, 0.785]])

dqn_10_sm = np.array([[0.4, 0.0, 0.8, 0.2, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.4, 0.8, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.6, 1.0, 0.6], [1.0, 0.0, 1.0, 1.0, 1.0, 0.4, 1.0, 0.0, 0.6, 0.6], [1.0, 0.0, 0.4, 0.8, 1.0, 0.6, 0.0, 1.0, 0.4, 0.0]])

bo_10_sm = np.array([[0.6, 0.8, 0.6, 0.2, 1.0, 0.8, 0.4, 0.0, 0.0, 0.2], [0.4, 0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0, 0.4, 0.0], [1.0, 0.6, 1.0, 0.0, 0.8, 0.6, 0.0, 0.0, 0.6, 1.0], [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.0, 0.6], [0.4, 0.6, 1.0, 1.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]])

bo_10_sp = np.array([[0.778, 0.784, 0.779, 0.798, 0.79, 0.776, 0.785, 0.772, 0.7809999999999999, 0.7759999999999999], [0.7699999999999999, 0.772, 0.764, 0.777, 0.768, 0.772, 0.766, 0.77, 0.788, 0.773], [0.783, 0.772, 0.7779999999999999, 0.783, 0.7819999999999999, 0.79, 0.7879999999999999, 0.785, 0.778, 0.79], [0.799, 0.777, 0.7779999999999999, 0.785, 0.768, 0.784, 0.776, 0.776, 0.78, 0.7819999999999998], [0.7889999999999999, 0.792, 0.775, 0.7729999999999999, 0.782, 0.788, 0.7790000000000001, 0.776, 0.773, 0.7769999999999999]])

bo_10_cm = np.array([[0.5980000000000001, 0.9675, 1.0470000000000002, 0.1925, 1.2195, 0.0, 0.018, 0.34049999999999997, 0.0, 0.521], [0.1715, 0.0, 0.8825, 0.513, 0.3225, 0.3675, 0.16899999999999998, 0.0, 0.20800000000000002, 1.0550000000000002], [1.721, 0.8515, 1.396, 1.203, 1.0245, 0.5065, 0.53, 0.3205, 1.5815, 1.722], [0.3435, 0.5305, 0.324, 0.8850000000000001, 0.0, 0.0, 1.0435, 0.525, 0.0, 0.8785000000000001], [1.342, 0.3205, 1.5875000000000001, 1.734, 0.44000000000000006, 1.565, 0.0, 0.6845, 0.0, 0.0]])

rs_10_sm = np.array([[0.6, 0.4, 0.4, 0.2, 0.8, 0.6, 0.0, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0, 0.6, 0.8, 0.2, 0.0, 1.0, 0.0], [1.0, 0.8, 1.0, 0.0, 0.8, 0.6, 0.0, 0.0, 0.8, 1.0], [0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.8, 0.6, 0.0, 0.4], [0.8, 0.4, 1.0, 1.0, 0.6, 1.0, 0.4, 0.0, 0.0, 0.0]])

rs_10_sp = np.array([[0.7729999999999999, 0.78, 0.778, 0.757, 0.7849999999999999, 0.781, 0.774, 0.7770000000000001, 0.783, 0.777], [0.77, 0.772, 0.771, 0.773, 0.8, 0.792, 0.767, 0.78, 0.78, 0.774], [0.7979999999999999, 0.789, 0.777, 0.777, 0.768, 0.7689999999999999, 0.783, 0.774, 0.7849999999999999, 0.78], [0.776, 0.7819999999999999, 0.7699999999999999, 0.7809999999999999, 0.782, 0.769, 0.779, 0.7770000000000001, 0.774, 0.776], [0.78, 0.775, 0.783, 0.7950000000000002, 0.78, 0.782, 0.7769999999999999, 0.775, 0.776, 0.775]])

rs_10_cm = np.array([[0.47400000000000003, 0.9324999999999999, 0.3785, 0.024, 1.212, 0.696, 0.0, 0.1765, 0.0, 0.0], [0.192, 0.0, 0.9950000000000001, 1.1960000000000002, 0.509, 0.374, 0.363, 0.0, 0.8240000000000001, 0.8425], [1.6789999999999998, 1.2375, 0.7205000000000001, 0.9075, 1.559, 0.37750000000000006, 0.316, 0.8460000000000001, 0.29900000000000004, 1.3705000000000003], [0.3545, 0.357, 0.33199999999999996, 0.0, 0.0, 0.0, 0.8665, 0.3625, 0.0, 0.327], [1.0580000000000003, 0.663, 1.5619999999999998, 1.6949999999999998, 0.6685000000000001, 1.7325, 0.0, 0.5575, 0.0, 0.0]])

ppo_10_cm = np.array([[0.0, 1.659, 0.0, 1.716, 1.738, 1.708, 1.704, 1.755, 1.743, 1.732], 
                      [0.344, 1.688, 1.755, 1.7, 1.7, 1.691, 1.215, 1.729, 1.703, 1.689], 
                      [0.522, 1.7, 1.703, 0.702, 1.718, 1.696, 1.679, 1.714, 1.744, 1.711], 
                      [1.755, 0.0, 0.0, 0.0, 0.331, 0.0, 0.324, 1.224, 0.0, 0.0], 
                      [1.005, 1.027, 1.738, 1.691, 1.748, 1.385, 1.561, 1.701, 1.74, 1.692]])

ppo_10_sp = np.array([[0.775, 0.765, 0.786, 0.792, 0.776, 0.769, 0.777, 0.776, 0.776, 0.787], 
                      [0.771, 0.781, 0.767, 0.777, 0.786, 0.777, 0.786, 0.789, 0.788, 0.771], [0.768, 0.791, 0.782, 0.773, 0.768, 0.774, 0.777, 0.773, 0.787, 0.782], [0.775, 0.764, 0.766, 0.784, 0.773, 0.793, 0.791, 0.772, 0.785, 0.795], [0.773, 0.779, 0.784, 0.79, 0.776, 0.788, 0.784, 0.782, 0.771, 0.783]])

ppo_10_sm = np.array([[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.8, 1.0, 0.0, 1.0], 
                      [0.0, 1.0, 1.0, 1.0, 0.6, 0.8, 0.0, 1.0, 0.2, 0.0], 
                      [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.0], 
                      [0.8, 0.2, 1.0, 0.4, 0.0, 1.0, 0.0, 0.8, 0.0, 0.0], 
                      [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])

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