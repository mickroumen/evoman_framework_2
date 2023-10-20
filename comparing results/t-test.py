import os
import pandas as pd
from scipy.stats import ttest_ind

ea_enemy_combinations = [
    {"EA": "EA_dynamic_comma.csv", "name": "EA dynamic comma"},
    {"EA": "EA_dynamic_plus.csv", "name": "EA dynamic plus"},
    {"EA": "EA_static_plus.csv", "name": "EA static plus"},
    {"EA": "EA_static_comma.csv", "name": "EA static comma"}
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
t_test_results = {}
for i in range(len(all_data)):
    for j in range(i + 1, len(all_data)):
        ea1_data, ea1_name = all_data[i]
        ea2_data, ea2_name = all_data[j]
        t_stat, p_value = ttest_ind(ea1_data, ea2_data, equal_var=False)  # Assuming unequal variances
        t_test_results[f"{ea1_name} vs {ea2_name}"] = {"t_statistic": t_stat, "p_value": p_value}
t_test_results