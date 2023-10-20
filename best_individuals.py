import csv
from baseline import EvoMan

def run_test_for_experiment(i):
    unique_experiment_name = f"experiment_adaptive_comma_{i + 1}"
    
    # Create object of your class
    env = EvoMan(experiment_name=unique_experiment_name, enemies=[1, 2, 3, 4, 5, 6, 7, 8], population_size=1, generations=1, mutation_rate=1, crossover_rate=1, mode='test', 
               n_hidden_neurons=10, headless=True, dom_l=-1, dom_u=1, speed='fastest', number_of_crossovers=1, n_elitism=1, k_tournament=1, type_of_selection_pressure='exponential', 
               k_tournament_final_linear_increase_factor=1, alpha=1, enemy_threshold=1, lamba_mu_ratio=1)

    # Call the test function
    results = env.test()
    return results

if __name__ == "__main__":
    all_results = [run_test_for_experiment(i) for i in range(10)]

    # Save the results to a CSV file
    with open("test_results.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Experiment', 'Fitness', 'Health Gain', 'Average Health Gain', 'Total Health Gain', 'Player Life', 'Enemy Life', 'Time'])
        
        # Write each result
        for i, result in enumerate(all_results, 1):
            if result:
                writer.writerow([f"experiment_adaptive_comma_bigger_{i}"] + list(result.values()))