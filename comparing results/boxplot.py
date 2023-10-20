import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 8))

ea_enemy_combinations = [
    {"EA": "EA_dynamic_comma.csv", "name": "Dynamic (\u03BC,\u03BB)"},
    {"EA": "EA_dynamic_plus.csv", "name": "Dynamic (\u03BC+\u03BB)"},
    {"EA": "EA_static_plus.csv", "name": "Static (\u03BC+\u03BB)"},
    {"EA": "EA_static_comma.csv", "name": "Static (\u03BC,\u03BB)"}
]

# Create an empty list to store the data
all_data = []

for combo in ea_enemy_combinations:
    ea_file = combo["EA"]
    ea_name = combo["name"]
    file_path = f"ResultsBoxplot40/{ea_file}"

    df = pd.read_csv(file_path)
    max_best_health = df["Average Health Gain"]
    
    # Add the data and label for this EA
    all_data.append((max_best_health, ea_name))

# Create a single boxplot with multiple categories and set colors
colors = ["blue", "blue", "grey", "grey"]  # Blue for Dynamic, Green for Static

ax = sns.boxplot(data=[data[0] for data in all_data], showfliers=False, palette=colors, boxprops=dict(alpha=0.5))
ax.set_xticklabels([data[1] for data in all_data], fontsize=15)  # Increase x-axis label font size
ax.yaxis.label.set_size(16)  # Increase y-axis label font size
plt.title("Average gains of different EAs", fontsize=20)  # Increase title font size

# Increase font size for y-axis values and set y-axis label
ax.tick_params(axis='y', labelsize=16)
ax.set_ylabel("Average gain tested on 8 enemies", fontsize=15)

plt.tight_layout()

plt.savefig("boxplots.png", dpi=300)

plt.show()
