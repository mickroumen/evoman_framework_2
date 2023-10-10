from generalist import run_evoman
import time

parameters = {
    'experiment_name': "testing_generalistx10",
    'enemy': [1,3,4,6],
    'population_size': 100,
    'generations': 30,
    'mutation_rate': 0.05,
    'crossover_rate': 0.3,
    'mode': "train",
    'number_of_crossovers' : 7, 
    'n_elitism': 7, 
    'k_tournament' : 13, 
    'sel_pres_incr': True, 
    'k_tournament_final_linear_increase_factor':3,
    'n_hidden_neurons': 10,
    'headless': True,
    'dom_l': -1,
    'dom_u': 1,
    'speed': "fastest", 
}


# for running and saving the code 10 times
for i in range(10):
    number = i + 1
    unique_experiment_name = f"experiment_{number}"
    parameters["experiment_name"] = unique_experiment_name
    run_evoman(**parameters)