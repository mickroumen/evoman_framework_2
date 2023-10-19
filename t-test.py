import os
import pandas as pd
from scipy.stats import ttest_ind

t_test_results = {}

enemies = [4, 6, 7]
EA = ["with_fitness_", "withouth_fitness_"]
      
for i, enemy in enumerate(enemies):
    max_values1 = []  
    max_values2 = [] 
    
    for j, ea in enumerate(EA):
        max_values_per_enemy = []  
        for k in range(1, 11):
            dir_name = f"resulsEAStest/{ea}{enemy}_{k}"
            file_path = os.path.join(dir_name, "results.csv")
            
            df = pd.read_csv(file_path)
            
            max_best_health = df["Best Health"].max()
            
            if ea == "with_fitness_":
                max_values1.append(max_best_health)
            else:
                max_values2.append(max_best_health)
        
    t_stat, p_value = ttest_ind(max_values1, max_values2)
    
    t_test_results[f"Enemy_{enemy}_EA_1_vs_EA_2"] = {"t_stat": t_stat, "p_value": p_value}

for key, value in t_test_results.items():
    print(f"{key}: t-statistic = {value['t_stat']}, p-value = {value['p_value']}")
