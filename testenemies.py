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

# Define the base file path
script_dir = os.path.dirname(os.path.abspath(__file__))

class EvoMan:
    def __init__(self, experiment_name, enemies, population_size, generations, mutation_rate, crossover_rate, 
                 mode, n_hidden_neurons, headless, dom_l, dom_u, speed, number_of_crossovers, n_elitism, k_tournament, type_of_selection_pressure, k_tournament_final_linear_increase_factor,
                 alpha, sigma_range):
        self.experiment_name = experiment_name
        self.enemies = enemies
        self.n_pop = population_size
        self.gens = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mode = mode
        self.n_hidden_neurons = n_hidden_neurons
        self.headless = headless
        self.dom_l = dom_l
        self.dom_u = dom_u
        self.number_of_crossovers = number_of_crossovers
        self.n_elitism = n_elitism
        self.k_tournament = k_tournament
        self.type_of_selection_pressure = type_of_selection_pressure
        self.k_tournament_final_linear_increase_factor = k_tournament_final_linear_increase_factor
        self.alpha = alpha
        self.current_generation = 0
        self.speed = speed
        self.counter = 0
        self.sigma_range = sigma_range
        
        # Setup directories
        self.setup_directories()

        # Set up the Evom   n environment
        self.env = self.initialize_environment()

        # Setup total network weights
        self.network_weights()
        self.tau_glob = 1/np.sqrt(2*self.total_network_weights)
        self.tau_ind = 1/np.sqrt(2*np.sqrt(self.total_network_weights))

    def setup_directories(self):
        results_dir = os.path.join(script_dir, "Results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        self.experiment_dir = os.path.join(results_dir, self.experiment_name)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def network_weights(self):
        self.n_inputs = self.env.get_num_sensors()
        self.n_outputs = 5

        self.total_network_weights = ((self.n_inputs + 1) * self.n_hidden_neurons + (self.n_hidden_neurons + 1) * self.n_outputs)

    def initialize_environment(self):
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        experiment_path = os.path.join("Results", self.experiment_name)
        
        # def fitness_single(self):
        #     return 0.5*(100 - self.get_enemylife()) + 0.5*self.get_playerlife() + np.log(self.get_time())
        # def fitness_single(self):
        #     win_count = sum([1 for playerlife, enemylife in zip(self.get_playerlife(), self.get_enemylife()) if playerlife - enemylife > 0])
        #     gains = [playerlife - enemylife for playerlife, enemylife in zip(self.get_playerlife(), self.get_enemylife())]
        #     average_gain = sum(gains) / len(gains)

        #     fitness = win_count + (0.02 * average_gain)

        #     return fitness

        def fitness_single(self):
            return 1
        # def cons_multi2(self,values):
        #     values = np.array([np.sqrt(value) if value > 0 else value for value in values])
        #     return values.mean()

        def cons_multi2(self, values):
            return values
        
        env = Environment(experiment_name=experiment_path,
                          enemies=self.enemies,
                          multiplemode="yes",
                          playermode="ai",
                          player_controller=player_controller(self.n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed=self.speed,
                          visuals=not self.headless)
        
        env.cons_multi = cons_multi2.__get__(env)
        env.fitness_single = fitness_single.__get__(env)
        return env
    
    def initialize_individual(self):
        # Initialize an individual with random weights and biases within the range [dom_l, dom_u]
        weights = np.random.uniform(self.dom_l, self.dom_u, self.total_network_weights)
        weights[213] = 100000
        sigmas = np.random.uniform(self.sigma_range[0], self.sigma_range[1], self.total_network_weights)
        individual = {
            'weights': weights,
            'sigmas': sigmas
        }
        return individual
    
    def simulation(self, x):
        self.counter += 1
        fitness, player_life, enemy_life, time = self.env.play(pcont=x)
        health_gain = player_life - enemy_life

        return fitness, health_gain, time, player_life, enemy_life
    
    def evaluate(self, population):
        # Evaluates the entire population and returns the fitness values
        fitness = np.zeros(self.n_pop)
        health_gain = np.zeros(self.n_pop)
        time_game = np.zeros(self.n_pop)
        player_life = np.zeros(self.n_pop)
        enemy_life = np.zeros(self.n_pop)
        for i, individual in enumerate(population):
            fitness[i], health_gain[i], time_game[i], player_life[i], enemy_life[i] = self.simulation(individual['weights'])
        return fitness, health_gain, time_game, player_life, enemy_life
    
    def mutate(self, individual):
        # Applies bit mutation to the individual based on the mutation rate
        for i in range(len(individual['weights'])):
            if random.uniform(0, 1) < self.mutation_rate:
                # Convert the weight to binary representation
                binary_representation = list(format(int((individual['weights'][i] - self.dom_l) * (2**15) / (self.dom_u - self.dom_l)), '016b'))
                
                for j in range(len(binary_representation)):
                    if random.uniform(0, 1) < self.mutation_rate:
                        binary_representation[j] = '1' if binary_representation[j] == '0' else '0'
                
                individual['weights'][i] = int("".join(binary_representation), 2) * (self.dom_u - self.dom_l) / (2**15) + self.dom_l
        return individual
    
    
    def mutate(self, individual):
        # Applies mutation to the individual based on the mutation rate
        #Mutate weights
        for i in range(len(individual['weights'])):
            if random.uniform(0, 1) < self.mutation_rate:
                mutation = np.random.normal(individual['weights'][i], individual['sigmas'][i]**2)
                individual['weights'][i] = np.clip(mutation, self.dom_l, self.dom_u)
        
        #Update sigmas
        update_factor_global = np.exp(np.random.normal(0,self.tau_glob**2))
        for i in range(len(individual['sigmas'])):
            update_factor_ind = np.random.normal(0,self.tau_ind**2)
            updating_factor = np.exp(update_factor_global + update_factor_ind)
            individual['sigmas'][i] *= updating_factor
            
        return individual 
    

    def crossover(self, parent1, parent2, number_of_crossovers):
        # Applies N point crossover
        child1 = parent1.copy()
        child2 = parent2.copy()            
        
        if random.uniform(0,1) < self.crossover_rate:
            crossover_points = sorted(random.sample(range(1, len(parent1)), number_of_crossovers))
            for i in range(number_of_crossovers - 1):
                # Switch between parents for each section
                child1[crossover_points[i]:crossover_points[i+1]] = parent2[crossover_points[i]:crossover_points[i+1]]
                child1[crossover_points[i]:crossover_points[i+1]] = parent1[crossover_points[i]:crossover_points[i+1]]
                
        return child1, child2
    
    # tournament (returns winnning individual and its fitness)
    def tournament_selection(self, candidate_indices, fitness, population, k=2):
        # Generate k unique random indices
        random_indices = np.random.choice(candidate_indices, k, replace=False)
        # Select the individuals with the highest fitness values among the randomly chosenpytrel ones
        best_individual_index = np.argmax(fitness[random_indices])
        # Return the index of the best individual on the population scale
        winner_index = random_indices[best_individual_index]

        #return the winning individual for parent selection and the winning index for tournament selection in the selection function
        return population[winner_index], winner_index
    
    def elitism(self, k, fitness):
        """
        Select the top k individuals
        """
        best_indices = np.argsort(fitness)[-k:]
        candidate_indices = np.arange(fitness.shape[0])
        for best_index in best_indices:
            candidate_indices = np.delete(candidate_indices, np.where(candidate_indices == best_index))

        return best_indices, candidate_indices


    # returns selected individuals and their fitness
    def selection(self, population, fitness):        
        #Now elitism function also deletes the elites from self.pop and self.pop_fit so they cant be chosen in the selection step
        if self.n_elitism > 0:
            selected_indices, candidate_indices = self.elitism(self.n_elitism, fitness)
        else:
            selected_indices = np.array([], dtype=int)
            candidate_indices = np.arange(fitness.shape[0])
        #if selection pressure is set to True we want the pressure to increase and thus the number of neural networks to compete to increase
        #eventually the number of individuals in tournament is doubled from start to finish.
        if self.type_of_selection_pressure=="linear":
            k = max(math.ceil(self.current_generation*self.k_tournament_final_linear_increase_factor/self.gens)*self.k_tournament, self.k_tournament)
        elif self.type_of_selection_pressure=="exponential":
            k_end = self.k_tournament_final_linear_increase_factor*self.k_tournament
            exp_rate = 1/self.gens
            k = self.k_tournament + math.ceil((k_end - self.k_tournament) * (1 - math.exp(-exp_rate * self.current_generation)))
        else:
            k = self.k_tournament
        for p in range(self.n_pop-self.n_elitism):
            winner, winner_index = self.tournament_selection(candidate_indices, fitness, population, k)
            # Remove the selected elite individuals and their fitness values from population and fitness
            selected_indices = np.append(selected_indices, winner_index)
            candidate_indices = np.delete(candidate_indices, np.where(candidate_indices == winner_index))

        return selected_indices
    
    def update_enemies(self, enemies, best_individual):
        #change the fitness and gain function so that it returns np.array of the fitness and gain for each enemy
        def cons_multi2(self,values):
            return values
        
        self.env.cons_multi = cons_multi2.__get__(self.env)

        # Run the simulation with the best individual
        fitnesses, health_gains, times, player_lifes, enemy_lifes = self.simulation(best_individual['weights'])

        enemies_to_remove = {enemy for enemy, health_gain in zip(enemies, health_gains) if health_gain > 40}
        remaining_enemies = [enemy for enemy in enemies if enemy not in enemies_to_remove]

        for _ in enemies_to_remove:
            while True:
                new_enemy = random.randint(1, 8)
                if new_enemy not in enemies:
                    remaining_enemies.append(new_enemy)
                    break
        
        return remaining_enemies
    
    def run(self):
        if self.mode == "train":
            fitness = self.train()

            return fitness
        elif self.mode == "test":
            self.test()
    
    def train(self):
        # take time of entire run
        start_time = time.time()

        # Initialize population
        population = np.array([self.initialize_individual() for _ in range(self.n_pop)])

        # Evaluate the initial population
        fitness, health_gain, time_game, player_life, enemy_life = self.evaluate(population)
        print(health_gain)
        raise

        # Initialize best individual and its fitness
        best_individual_index = np.argmax(fitness)
        best_individual = population[best_individual_index]
        best_fitness = fitness[best_individual_index]

        # Creat csv file
        results_file_path = os.path.join(self.experiment_dir, "results.csv")
        with open(results_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Best Fitness", "Average Fitness", "Std Fitness", "Best Health",
                             "Avg Health", "Std Health", "Lowest Time", "Avg Time", "Std Time"])
            
                        # Save initial population
            writer.writerow([0, np.max(fitness), np.mean(fitness), np.std(fitness),
                    np.max(health_gain), np.mean(health_gain), np.std(health_gain),
                    np.min(time_game), np.mean(time_game), np.std(time_game)])

            # Main loop for generations
            for gen in range(self.gens):
                children = []
                self.current_generation += 1
                for _ in range(self.n_pop // 2):  # Two children per iteration
                    parent1, winner_index = self.tournament_selection(population.shape[0], fitness, population)
                    parent2, winner_index = self.tournament_selection(population.shape[0], fitness, population)

                    child1 = parent1.copy()
                    child2 = parent2.copy()            
                    child1['weights'], child2['weights'] = self.crossover(parent1['weights'], parent2['weights'], self.number_of_crossovers)

                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                    children.extend([child1, child2])    
                fitness_children, health_gain_children, time_game_children, player_life_children, enemy_life_children  = self.evaluate(children)
                
                pop_before_selection = np.concatenate((population, children))
                fitness_before_selection = np.concatenate((fitness, fitness_children))
                #Perform selection on the combined population of parents and children
                selected_indices = self.selection(pop_before_selection, fitness_before_selection)

                health_gain_before_selection = np.concatenate((health_gain, health_gain_children))
                time_game_before_selection = np.concatenate((time_game, time_game_children))
                fitness = fitness_before_selection[selected_indices]
                population = pop_before_selection[selected_indices]
                health_gain = health_gain_before_selection[selected_indices]
                time_game = time_game_before_selection[selected_indices]

                # Check if any individual has a higher fitness, save that one
                max_fitness_index = np.argmax(fitness)
                if fitness[max_fitness_index] > best_fitness:
                    best_fitness = fitness[max_fitness_index]
                    best_individual = population[max_fitness_index]

                # save to csv
                writer.writerow([gen, np.max(fitness), np.mean(fitness), np.std(fitness),
                                    np.max(health_gain), np.mean(health_gain), np.std(health_gain),
                                    np.min(time_game), np.mean(time_game), np.std(time_game)])
                
                print(self.enemies)
                # change enemies if met threshold
                self.enemies = self.update_enemies(self.enemies, best_individual)
                self.env.enemies = self.enemies
                print(self.enemies)

                def cons_multi2(self,values):
                    values = np.array([np.sqrt(value) if value > 0 else value for value in values])
                    return values.mean()
                
                self.env.cons_multi = cons_multi2.__get__(self.env)
                
                print(f"Generation {gen}, Best Fitness: {np.max(fitness)} and index {np.argmax(fitness)}")
                print(f"Generation {gen}, Best Health: {np.max(health_gain)} and index {np.argmax(health_gain)}")               
                
                # Save the best individual's neural network weights
                np.save(os.path.join(self.experiment_dir, "best_individual.npy"), best_individual['weights'])

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Save the elapsed time
        time_file_path = os.path.join(self.experiment_dir, "training_time_and_effort.txt")
        with open(time_file_path, 'w') as file:
            file.write(f"Training Time: {elapsed_time} seconds\n")
            file.write(f"Tuning effort: {self.counter} \n")
        
        return best_fitness

    def test(self):
        best_individual_path = os.path.join(self.experiment_dir, "best_individual.npy")

        if os.path.exists(best_individual_path):
            # Load the best individual's neural network weights
            best_individual = np.load(best_individual_path)
            
            #change the fitness and gain function so that it returns np.array of the fitness and gain for each enemy
            def cons_multi2(self,values):
                return values

            self.env.cons_multi = cons_multi2.__get__(self.env)

            # Run the simulation with the best individual
            fitness, health_gain, time, player_life, enemy_life = self.simulation(best_individual)
            print(f'fitness is: {fitness}')
            print(f'health gain is: {health_gain}')
            print(f'player life is: {player_life}')
            print(f'enemy life is: {enemy_life}')
            print(f'time is: {time}')

        else:
            print("No best individual found!")


def run_evoman(experiment_name, enemies, population_size, generations, mutation_rate, crossover_rate, mode, 
               n_hidden_neurons, headless, dom_l, dom_u, speed, number_of_crossovers, n_elitism, k_tournament, type_of_selection_pressure, k_tournament_final_linear_increase_factor, alpha, sigma_range):
        evoman = EvoMan(experiment_name, enemies, population_size, generations, mutation_rate, crossover_rate, 
                        mode, n_hidden_neurons, headless, dom_l, dom_u, speed, number_of_crossovers, n_elitism, k_tournament, type_of_selection_pressure, k_tournament_final_linear_increase_factor, alpha, sigma_range)
        
        # Log the command
        if mode == "train":
            log_file_path = os.path.join(evoman.experiment_dir, "commands_log.txt")
            with open(log_file_path, "a") as f:
                f.write(' '.join(sys.argv) + '\n')
        
        fitness = evoman.run()
        
        return fitness
        

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Evolutionary Algorithm for EvoMan")
        
        parser.add_argument("--experiment_name", type=str, default="experiment_enemy=6_k_increase=3", help="Name of the experiment")
        parser.add_argument("--enemies", type=int, nargs='+',default=[1, 2, 3, 4, 5, 6, 7, 8], help="Enemies numbers")
        parser.add_argument("--npop", type=int, default=100, help="Size of the population")
        parser.add_argument("--gens", type=int, default=30, help="Number of generations")
        parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate")
        parser.add_argument("--crossover_rate", type=float, default=0.5, help="Crossover rate")
        parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")
        parser.add_argument("--n_hidden_neurons", type=int, default=10, help="Number of hidden neurons")
        parser.add_argument("--headless", action="store_true", help="Run in headless mode")
        parser.add_argument("--dom_l", type=float, default=-1, help="Lower bound for initialization and mutation")
        parser.add_argument("--dom_u", type=float, default=1, help="Upper bound for initialization and mutation")
        parser.add_argument("--speed", type=str, default="fastest", help="Speed: fastest or normal")
        parser.add_argument("--number_of_crossovers", type=int, default=3, help="Number of crossovers")
        parser.add_argument("--n_elitism", type=int, default=2, help="Number of best individuals from population that are always selected for the next generation.")
        parser.add_argument("--k_tournament", type=int, default= 2, help="The amount of individuals to do a tournament with for selection, the more the higher the selection pressure")
        parser.add_argument("--type_of_selection_pressure", type=str, default="linear", help="if set to linear the selection pressure will linearly increase over time from k_tournament till k_tournament_final_linear_increase_factor*k_tournament, if set to exponential the selection pressure will increase exponentially from k_tournament till 2*k_tournament, if set to anything else the selection pressure will stay the same")
        parser.add_argument("--k_tournament_final_linear_increase_factor", type=int, default= 4, help="The factor with which k_tournament should linearly increase (if type_of_selection_pressure = True), if the value is 4 the last quarter of generations have tournaments of size k_tournament*4")
        parser.add_argument("--alpha", type=float, default=0.5, help="Weight for enemy damage")
        parser.add_argument("--sigma_range", type=list, default=[0.01,0.3], help="Range for sigma")
        
        args = parser.parse_args()

        run_evoman(args.experiment_name, args.enemies, args.npop, args.gens, args.mutation_rate, args.crossover_rate,
               args.mode, args.n_hidden_neurons, args.headless, args.dom_l, args.dom_u, args.speed, args.number_of_crossovers,
               args.n_elitism, args.k_tournament, args.type_of_selection_pressure, args.k_tournament_final_linear_increase_factor,
               args.alpha, args.sigma_range)