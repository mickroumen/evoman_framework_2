from EAadaptive_comma2 import run_evoman
from joblib import Parallel, delayed

parameters = {
    'experiment_name': "variable_enemies_commma",
    'enemies': [1, 4, 6, 7],
    'population_size': 200,
    'generations': 40,
    'mutation_rate': 0.054128345043897554,
    'crossover_rate': 1,
    'mode': "train",
    'n_hidden_neurons': 10,
    'headless': True,
    'dom_l': -1,
    'dom_u': 1,
    'speed': "fastest", 
    'number_of_crossovers': 2,
    'n_elitism': 2,
    'k_tournament': 4,
    'k_tournament_final_linear_increase_factor': 2,   
    'alpha': 0.5,
    'type_of_selection_pressure': 'exponential',
    'enemy_threshold': 10,
    'lamba_mu_ratio': 6
}


# for running and saving the code 10 times
def run_single_experiment(i):
    number = i + 1
    unique_experiment_name = f"experiment_adaptive_comma_bigger_{number}"
    parameters["experiment_name"] = unique_experiment_name
    run_evoman(**parameters)

if __name__ == "__main__":
    Parallel(n_jobs=-1)(delayed(run_single_experiment)(i) for i in range(10))