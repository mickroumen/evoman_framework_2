from testenemies import run_evoman

parameters = {
    'experiment_name': "test",
    'enemies': [2,5,7],
    'npop': 100,
    'gens': 30,
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
    'type_of_selection_pressure': 'linear',
    'k_tournament_final_linear_increase_factor': 4, 
    'alpha': 0.5,
    'sigma_range': [0.01, 0.5]  
}

run_evoman(**parameters)