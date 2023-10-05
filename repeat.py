from evoman_with_alpha import run_evoman
import time

parameters = {
    'experiment_name': "first_generalist",
    'enemies': [1, 3, 4, 6, 7],
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
    'alpha': 0.5,
}


# for running and saving the code 10 times
for i in range(10):
    number = i + 1
    unique_experiment_name = f"second_generalist_{number}"
    parameters["experiment_name"] = unique_experiment_name
    run_evoman(**parameters)