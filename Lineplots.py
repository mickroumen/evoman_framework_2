import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))

# Define the combinations of EA and sets of enemies
ea_enemy_combinations = [
    {"EA": "EA1", "enemy_set": 1},
    {"EA": "EA1", "enemy_set": 2},
    {"EA": "EA2", "enemy_set": 1},
    {"EA": "EA2", "enemy_set": 2},
]

for idx, combo in enumerate(ea_enemy_combinations):
    t = combo["EA"]
    enemy_set = combo["enemy_set"]

    concatenated_df = pd.DataFrame()

    for i in range(1, 11):
        dir_name = f"Results/experiment_{i}"
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

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"{t} with Enemy Set {enemy_set}")

    ax.legend()

plt.tight_layout()

plt.show()
