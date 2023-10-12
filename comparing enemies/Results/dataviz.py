import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Read the CSV data
data = pd.read_csv('comparing enemies/Results/results.csv')

# Step 2: Pivot the data to create a matrix
pivot_data = data.pivot(index='Training_Enemy', columns='Testing_Enemy', values='Fitness')

# Step 3: Create the matrix plot (heatmap) for Fitness
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.1f', cbar=True, square=True, linewidths=0.5)
plt.title('Fitness Matrix Plot')
plt.xlabel('Testing Enemy')
plt.ylabel('Training Enemy')
plt.show()

# Repeat Steps 2 and 3 for the Gain matrix
pivot_data_gain = data.pivot(index='Training_Enemy', columns='Testing_Enemy', values='Gain')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_data_gain, annot=True, cmap='YlGnBu', fmt='.1f', cbar=True, square=True, linewidths=0.5)
plt.title('Gain Matrix Plot')
plt.xlabel('Testing Enemy')
plt.ylabel('Training Enemy')
plt.show()
