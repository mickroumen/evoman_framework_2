from EA_static_comma import run_evoman
import time

parameters = {
    'experiment_name': "testing_generalistx10",
    'enemies': [1,4,6,7],
    'population_size': 100,
    'generations': 30,
    'mutation_rate': 0.236,
    'crossover_rate': 1,
    'mode': "train",
    'number_of_crossovers' : 8, 
    'n_elitism': 4, 
    'k_tournament' : 2, 
    'k_tournament_final_linear_increase_factor':4,
    'n_hidden_neurons': 10,
    'headless': True,
    'dom_l': -1,
    'dom_u': 1,
    'speed': "fastest", 
    'type_of_selection_pressure' : 'exponential',
    'alpha' : 0.5,
    'enemy_threshold': 15,
     'lamba_mu_ratio': 3
}


# for running and saving the code 10 times
for i in range(10):
    number = i + 1
    unique_experiment_name = f"experiment_{number}"
    parameters["experiment_name"] = unique_experiment_name
    run_evoman(**parameters)