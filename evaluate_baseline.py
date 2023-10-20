import pandas as pd

df = pd.read_csv("Results/beseline_test/baseline.csv")

average_enemies_beaten = df["EnemiesBeaten"].mean()
std_dev_enemies_beaten = df["EnemiesBeaten"].std()
max_enemies_beaten = df["EnemiesBeaten"].max()

enemies_beaten_counts = df["EnemiesBeaten"].value_counts()
best_count = enemies_beaten_counts[max_enemies_beaten]

print(f"Average Enemies Beaten: {average_enemies_beaten:.2f}")
print(f"Standard Deviation of Enemies Beaten: {std_dev_enemies_beaten:.2f}")
print(f"Maximum Enemies Beaten: {max_enemies_beaten}")
print(f"Number of occurrences of the best result: {best_count}")