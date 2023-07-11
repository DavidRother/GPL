import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# List of environments and algorithms
environments = ['Wolfpack', 'LBF', 'Ext-LBF', 'CookingZoo']
algorithms = ['MoOP-CA', 'MoOP-FR', 'GPL', 'MAPPO']
num_runs = 10
time_steps = 160000

# Generate synthetic data
data = []

for env in environments:
    for algo in algorithms:
        for run in range(num_runs):
            for time_step in range(0, time_steps + 1, 1000):  # assuming evaluations every 1000 steps
                return_value = np.random.normal(loc=(run+1) * 100, scale=20)  # example return value
                data.append([env, algo, run, time_step, return_value])

# Convert data to a pandas DataFrame
columns = ['environment', 'algorithm', 'run', 'time_step', 'return']
df = pd.DataFrame(data, columns=columns)

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Flatten the axis array for easier indexing
axs = axs.flatten()

# Loop over each environment
for i, environment in enumerate(environments):
    # Select data for the current environment
    data = df[df['environment'] == environment]

    # Loop over each algorithm
    for algorithm in algorithms:
        # Select data for the current algorithm
        algo_data = data[data['algorithm'] == algorithm]

        # Calculate mean and 95% confidence interval
        mean = algo_data.groupby('time_step')['return'].mean()
        ci = 1.96 * algo_data.groupby('time_step')['return'].std() / np.sqrt(10)  # for 95% CI

        # Plot mean with CI as shaded area
        axs[i].plot(mean.index, mean, label=algorithm)
        axs[i].fill_between(mean.index, (mean - ci), (mean + ci), alpha=0.1)

    # Set title and labels
    axs[i].set_title(f'Environment: {environment}')
    axs[i].set_xlabel('Time Step')
    axs[i].set_ylabel('Average Return')

# Add legend outside of the plots
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
