# Import necessary libraries
import math
import numpy as np
import argparse
from evoman.environment import Environment
import os
from demo_controller import player_controller
import csv
import random
import sys
import time
import numpy as np
import os
import csv
import numpy as np
from pyDOE2 import lhs
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from Bitflip_3_point_elitism import run_evoman
import csv
import os


# Define the parameters for the experiment
parameters = {
    'experiment_name': "comparing enemies",
    'enemy': [1],
    'population_size': 100,
    'generations': 30,
    'mutation_rate': 0.058178978426563575,
    'crossover_rate': 0.2574110793458289,
    'number_of_crossovers': 7,
    'n_elitism': 7,
    'k_tournament': 13,
    'k_tournament_final_linear_increase_factor': 3,
    'mode': "train",
    'n_hidden_neurons': 10,
    'headless': True,
    'dom_l': -1,
    'dom_u': 1,
    'speed': "fastest",
    'sel_pres_incr': True,
}

ennemies = [1,2,3,4,5,6,7,8]

# Create a results directory if it doesn't exist
results_dir = os.path.join(parameters['experiment_name'], "Results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Initialize a CSV file to store the results
results_file_path = os.path.join(results_dir, "results.csv")
with open(results_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Training_Enemy", "Testing_Enemy", "Fitness", "Gain","Time"])

# Loop through training and testing on different enemies
for train_enemy in ennemies:
    for test_enemy in ennemies:
        
            # Update the parameters for training and testing
            parameters['enemy'] = [train_enemy]
            parameters['mode'] = "train"

            # Run training on the current enemy
            best_fitness = run_evoman(**parameters)

            # Update the parameters for testing
            parameters['enemy'] = [test_enemy]
            parameters['mode'] = "test"

            # Run testing on the other enemy
            fitness, health_gain, timeu = run_evoman(**parameters)

            # Append the results to the CSV file
            with open(results_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([train_enemy, test_enemy, fitness,health_gain, timeu])

print("Training and testing completed. Results are saved in", results_file_path)
