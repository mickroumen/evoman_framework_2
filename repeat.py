from new_specialist_last import run_evoman
import time

parameters = {
    'experiment_name': "testing_specialistx10",
    'enemy': [6],
    'population_size': 100,
    'generations': 50,
    'mutation_rate': 0.1,
    'crossover_rate': 0.9,
    'tournament_size': 5,
    'mode': "train",
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