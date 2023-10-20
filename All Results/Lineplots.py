import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))

# Define the combinations of EA and sets of enemies
ea_enemy_combinations = [
    {"EA": "EA_dynamic_comma", "name": "EA dynamic enemyset using (\u03BC,\u03BB)"},
    {"EA": "EA_dynamic_plus", "name": "EA dynamic enemyset using (\u03BC+\u03BB)"},
    {"EA": "EA_static_comma", "name": "EA static enemyset [1,3,7,8] using (\u03BC,\u03BB)"},
    {"EA": "EA_static_plus", "name": "EA static enemyset [1,3,7,8] using (\u03BC+\u03BB)"},
]

for idx, combo in enumerate(ea_enemy_combinations):
    EA = combo["EA"]
    EA_name = combo["name"]
    concatenated_df = pd.DataFrame()

    for i in range(1, 11):
        dir_name = f"ResultsLine40/{EA}/experiment_{i}"
        file_path = os.path.join(dir_name, "results.csv")

        df = pd.read_csv(file_path)

        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)

    grouped_by_generation = concatenated_df.groupby("Generation")

    best_fitness_means = grouped_by_generation["Best Fitness"].mean()
    avg_fitness_means = grouped_by_generation["Average Fitness"].mean()

    upper_best_bounds_1sigma = best_fitness_means + grouped_by_generation["Std Fitness"].mean()
    lower_best_bounds_1sigma = best_fitness_means - grouped_by_generation["Std Fitness"].mean()

    upper_best_bounds_2sigma = best_fitness_means + 2 * grouped_by_generation["Std Fitness"].mean()
    lower_best_bounds_2sigma = best_fitness_means - 2 * grouped_by_generation["Std Fitness"].mean()

    upper_avg_bounds_1sigma = avg_fitness_means + grouped_by_generation["Average Fitness"].std()
    lower_avg_bounds_1sigma = avg_fitness_means - grouped_by_generation["Average Fitness"].std()

    upper_avg_bounds_2sigma = avg_fitness_means + 2 * grouped_by_generation["Average Fitness"].std()
    lower_avg_bounds_2sigma = avg_fitness_means - 2 * grouped_by_generation["Average Fitness"].std()

    # Determine the subplot location
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    ax.plot(
        best_fitness_means,
        label="Best Fitness",
        color="orange",
        linestyle='-'
    )
    ax.plot(
        avg_fitness_means,
        label="Average Fitness",
        color="blue",
        linestyle='-'
    )
    ax.fill_between(
        range(len(best_fitness_means)),
        lower_best_bounds_1sigma,
        upper_best_bounds_1sigma,
        color="lightblue",
        alpha=0.5
    )
    ax.fill_between(
        range(len(best_fitness_means)),
        lower_best_bounds_2sigma,
        upper_best_bounds_2sigma,
        color="lightblue",
        alpha=0.3
    )

    ax.fill_between(
        range(len(avg_fitness_means)),
        lower_avg_bounds_1sigma,
        upper_avg_bounds_1sigma,
        color="lightgrey",
        alpha=0.7
    )
    ax.fill_between(
        range(len(avg_fitness_means)),
        lower_avg_bounds_2sigma,
        upper_avg_bounds_2sigma,
        color="lightgrey",
        alpha=0.3
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"{EA_name}")

    # Increase font size for various text elements
    ax.set_title(f"{EA_name}", fontsize=20)  # Title font size
    ax.set_xlabel("Generation", fontsize=18)  # X-axis label font size
    ax.set_ylabel("Fitness", fontsize=18)  # Y-axis label font size

    ax.legend(fontsize=12)  # Legend font size

    # Increase font size for tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=16)  # X-axis tick labels font size
    ax.tick_params(axis='y', labelsize=16)  # Y-axis tick labels font size

plt.tight_layout()
plt.savefig("Lineplots30gen.jpg", dpi=500)
plt.show()
