import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('all_performances.csv')
df = pd.DataFrame(data)

max_fitness_index = df['Fitness'].idxmax()
max_alpha = df['alpha'][max_fitness_index]
max_fitness = df['Fitness'][max_fitness_index]

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='alpha', y='Fitness', color='blue')

plt.xlim(0, 1)
plt.xticks(np.arange(0, 1.1, 0.1))

plt.ylim(df['Fitness'].min(),100)

plt.annotate(f'\u03B1: {max_alpha:.2f}', (max_alpha, df['Fitness'].min()), textcoords="offset points", xytext=(0, 10),
             ha='right', fontsize=10, color='red')

plt.plot([max_alpha, max_alpha], [max_fitness, df['Fitness'].min()], color='red', linestyle='--')

plt.xlabel('Alpha')
plt.ylabel('Fitness')
plt.title('Fitness vs. Alpha')
plt.savefig("alphavsfitness.png", dpi=300)

plt.show()
