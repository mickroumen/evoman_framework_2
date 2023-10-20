from EA_staticEnemies_plus import run_evoman
from joblib import Parallel, delayed

parameters = {
    'experiment_name': "testingEA-AdaptiveEnemies_comma",
    'enemies': [1,  3, 7, 8],
    'population_size': 100,
    'generations': 30,
    'mutation_rate': 0.5,
    'crossover_rate': 0.8,
    'mode': "train",
    'n_hidden_neurons': 10,
    'headless': True,
    'dom_l': -1,
    'dom_u': 1,
    'speed': "fastest", 
    'number_of_crossovers': 10,
    'n_elitism': 2,
    'k_tournament': 1,
    'k_tournament_final_linear_increase_factor': 10,   
    'alpha': 0.5,
    'type_of_selection_pressure': 'exponential',
    'enemy_threshold': 10,
    'lamba_mu_ratio': 6
}


# for running and saving the code 10 times
def run_single_experiment(i):
    number = i + 1
    unique_experiment_name = f"experiment_testing_{number}"
    parameters["experiment_name"] = unique_experiment_name
    run_evoman(**parameters)

if __name__ == "__main__":
    Parallel(n_jobs=-1)(delayed(run_single_experiment)(i) for i in range(10))