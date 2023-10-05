import numpy as np
from pyDOE2 import lhs
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from Bitflip_3_point_elitism import run_evoman
import csv
import os


# Define the main directory to store the results
main_dir = 'Results_Tuning'

# Create directories if they don't exist
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

# Create subdirectories for average and individual performances
average_dir = os.path.join(main_dir, 'averages')
individual_dir = os.path.join(main_dir, 'individuals')

if not os.path.exists(average_dir):
    os.makedirs(average_dir)

if not os.path.exists(individual_dir):
    os.makedirs(individual_dir)


# Define fixed parameters and parameters to change
parameters = {
    'experiment_name': "",
    'enemy': [6],
    'population_size': 100,
    'generations': 30,
    'mutation_rate': 0.1,
    'crossover_rate': 0.5,
    'mode': "train",
    'n_hidden_neurons': 10,
    'headless': True,
    'dom_l': -1,
    'dom_u': 1,
    'speed': "fastest", 
    'number_of_crossovers': 3,
    'n_elitism': 2,
    'k_tournament': 4,
    'sel_pres_incr': True,
    'k_tournament_final_linear_increase_factor': 4,   
}

param_change = {
    'mutation_rate': {'range': (0.01, 0.2), 'type': 'float'},
    'crossover_rate': {'range': (0.2, 1), 'type': 'float'},
    'n_hidden_neurons': {'range': (4, 20), 'type': 'int'},
    'number_of_crossovers': {'range': (2, 10), 'type': 'int'},
    'n_elitism': {'range': (1, 20), 'type': 'int'},
    'k_tournament': {'range': (2, 20), 'type': 'int'},
    'k_tournament_final_linear_increase_factor': {'range': (2, 5), 'type': 'int'}
}


n_samples = 10
n_iterations = 10
n_runs = 10

evaluated_parameters = []
average_performances = []
all_performances = []

# Normalize ranges and generate initial design points
ranges = np.array([param['range'] for param in param_change.values()])
normalized_ranges = np.array([[0, 1]] * len(ranges))
initial_design = lhs(len(ranges), samples=n_samples)
scaled_design = initial_design * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]

# Evaluate initial design points and fit the model
initial_performance = []
for point in scaled_design:
    varying_parameters = dict(zip(param_change.keys(), point))
    combined_parameters = {**parameters, **varying_parameters}
    for param_name, param_properties in param_change.items():
        if param_properties['type'] == 'int':
            combined_parameters[param_name] = int(round(combined_parameters[param_name]))

    performance_runs = [run_evoman(**combined_parameters) for _ in range(n_runs)]
    average_performance = np.mean(performance_runs)

    evaluated_parameters.append(varying_parameters)
    initial_performance.append(average_performance)
    all_performances.append(performance_runs)
    average_performances.append(average_performance)

model = GaussianProcessRegressor()
model.fit(initial_design, np.array(initial_performance))

# Optimize and update the model iteratively
for iteration in range(n_iterations):
    def objective(x):
        return -model.predict([x])[0]
    
    res = minimize(objective, [0.5] * len(ranges), bounds=normalized_ranges)
    next_point = res.x * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]

    varying_parameters = dict(zip(param_change.keys(), next_point))
    for param_name, param_properties in param_change.items():
        if param_properties['type'] == 'int':
            varying_parameters[param_name] = int(round(varying_parameters[param_name]))

    combined_parameters = {**parameters, **varying_parameters}
    performance_runs = [run_evoman(**combined_parameters) for _ in range(n_runs)]
    next_performance = np.mean(performance_runs)

    evaluated_parameters.append(varying_parameters)
    average_performances.append(next_performance)
    all_performances.append(performance_runs)
    model.fit(np.vstack([model.X_train_, res.x]), np.hstack([model.y_train_, next_performance]))

# Find the best parameters
best_index = np.argmax(average_performances)
best_parameters = evaluated_parameters[best_index]

print(f"Best Parameters: {best_parameters}")

with open(os.path.join(average_dir, 'average_performances.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(list(param_change.keys()) + ['Average Performance'])
    for params, avg_performance in zip(evaluated_parameters, average_performances):
        writer.writerow(list(params.values()) + [avg_performance])

with open(os.path.join(individual_dir, 'all_performances.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(list(param_change.keys()) + ['Run ' + str(i) for i in range(1, n_runs + 1)])
    for params, performances in zip(evaluated_parameters, all_performances):
        writer.writerow(list(params.values()) + performances)

with open(os.path.join(individual_dir, 'best_parameters.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(best_parameters.keys())
    writer.writerow(best_parameters.values())