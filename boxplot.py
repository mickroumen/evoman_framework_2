import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

enemies = [4, 6, 7]
EA = ["with_fitness_", "withouth_fitness_"]

y_limits = [(40, 85), (25, 70), (45, 100)]  

plt.figure(figsize=(15, 8))  

for i, enemy in enumerate(enemies):
    max_values = []  
    labels = []  
    for j, ea in enumerate(EA):
        max_values_per_enemy = [] 
        for k in range(1, 11):
            dir_name = f"resulsEAStest/{ea}{enemy}_{k}"
            file_path = os.path.join(dir_name, "results.csv")

            df = pd.read_csv(file_path)

            max_best_health = df["Best Health"].max()

            max_values_per_enemy.append(max_best_health)

            labels.append(
                f"{'Optimized Fitness' if 'with_fitness' in ea else 'Normal Fitness'}"
            )

        max_values.extend(max_values_per_enemy)

    plt.subplot(1, len(enemies), i + 1) 

    sns.boxplot(
        x=labels,
        y=max_values,
        showfliers=False,  
        color="lightblue", 
        boxprops=dict(alpha=0.7)  
    )

    plt.ylabel("Individual Gain")
    plt.title(f"EAs - Enemy {enemy}")

    plt.ylim(y_limits[i])

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)  
plt.savefig("boxplots.png", dpi=300)

plt.show()
