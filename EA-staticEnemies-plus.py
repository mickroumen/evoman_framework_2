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
                 alpha, enemy_threshold,lamba_mu_ratio):
        self.experiment_name = experiment_name
        self.enemies = enemies
        self.n_pop = population_size
        self.gens = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mode = mode
        self.n_hidden_neurons = n_hidden_neurons
        self.headless = True
        self.dom_l = dom_l
        self.dom_u = dom_u
        self.number_of_crossovers = number_of_crossovers
        self.n_elitism = n_elitism
        self.k_tournament_start = k_tournament
        self.k_tournament = k_tournament
        self.type_of_selection_pressure = type_of_selection_pressure
        self.k_tournament_final_linear_increase_factor = k_tournament_final_linear_increase_factor
        self.alpha = alpha
        self.current_generation = 0
        self.speed = speed
        self.counter = 0
        self.enemy_threshold = enemy_threshold
        self.updated_enemies = False
        self.lamba_mu_ratio = lamba_mu_ratio

        # Setup directories
        self.setup_directories()

        # Set up the Evoman environment
        self.env = self.initialize_environment()

        # Setup total network weights
        self.network_weights()

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
        
        def fitness_single(self):
             return 1
        
        #def cons_multi2(self,values):
             #values = np.array([value**0.8 if value > 0 else value for value in values])
             #return values.mean()
                        
        def cons_multi2(self,values):
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
        individual = np.random.uniform(self.dom_l, self.dom_u, self.total_network_weights)
        # Add the mutation rate as the last element of the individual
        individual = np.append(individual, 0.5)
        # Make sure the player is always shooting
        individual[213] = 10000
        
        return individual
    
    def simulation(self, x):
        self.counter += 1
        fitness, player_life, enemy_life, time = self.env.play(pcont=x)
        
        health_gain = player_life - enemy_life

        return fitness, health_gain, time, player_life, enemy_life
    
    def evaluate(self, population):
        fitness = np.empty(population.shape[0], dtype=object)
        health_gain = np.empty(population.shape[0], dtype=object)
        time_game = np.empty(population.shape[0], dtype=object)
        player_life = np.empty(population.shape[0], dtype=object)
        enemy_life = np.empty(population.shape[0], dtype=object)
        for i, individual in enumerate(population):
            fitness[i], health_gain[i], time_game[i], player_life[i], enemy_life[i] = self.simulation(individual[:-1])
        return fitness, health_gain, time_game, player_life, enemy_life

    """
    def mutate(self, individual):
        # Applies bit mutation to the individual based on the mutation rate
        if random.uniform(0, 1) < self.mutation_rate:
            for i in range(len(individual[:-1])):            
                # Convert the weight to binary representation
                binary_representation = list(format(int((individual[i] - self.dom_l) * (2**15) / (self.dom_u - self.dom_l)), '016b'))
                
                for j in range(len(binary_representation)):
                    
                        binary_representation[j] = '1' if binary_representation[j] == '0' else '0'
                
                individual[i] = int("".join(binary_representation), 2) * (self.dom_u - self.dom_l) / (2**15) + self.dom_l
        return individual
    """

    def mutate(self, individual):
        # Applies mutation to the individual based on the mutation rate
        if random.uniform(0, 1) < self.mutation_rate:
            #Update sigma
            tau = 1 / np.sqrt(len(individual[:-1]))
            individual[-1] = individual[-1] * np.exp(tau * np.random.normal())
            individual[-1] = np.clip(individual[-1], 0.01, 0.9)
            for i in range(len(individual[:-1])):
                individual[i] += individual[-1] * np.random.normal()
        return individual       
    
    """
    def crossover(self, parent1, parent2, n):
            # Applies the uniform crossover operator
            if random.uniform(0,1) < self.crossover_rate:
                child1 = np.zeros(len(parent1))
                child2 = np.zeros(len(parent1))
                for i in range(len(parent1)):
                    if random.uniform(0,1) < 0.5:
                        child1[i] = parent1[i]
                        child2[i] = parent2[i] 
                    else:
                        child1[i] = parent2[i]
                        child2[i] = parent1[i]
                return child1, child2
            else:
                return parent1.copy(), parent2.copy()
    """        
    
    def crossover(self, parent1, parent2, number_of_crossovers):
        # Applies N point crossover
        # if random.uniform(0,1) < self.crossover_rate: 
            crossover_points = sorted(random.sample(range(1, len(parent1)), number_of_crossovers))
            child1 = parent1.copy()
            child2 = parent2.copy()
            if random.uniform(0,1) < self.crossover_rate: 
                crossover_points = sorted(random.sample(range(1, len(parent1 -1)), number_of_crossovers))
                crossover_points.append(-2)            
                
                for i in range(number_of_crossovers):
                # Switch between parents for each section
                    if i%2 != 0:
                        child1[crossover_points[i-1]:crossover_points[i]] = parent2[crossover_points[i-1]:crossover_points[i]]
                        child2[crossover_points[i-1]:crossover_points[i]] = parent1[crossover_points[i-1]:crossover_points[i]]
            return child1, child2
    
    # tournament (returns winnning individual and its fitness)
    def tournament_selection_parents(self, candidate_indices, fitness):
        # Generate k unique random indices
        random_indices = np.random.choice(candidate_indices, self.k_tournament, replace=False)
        
        best_individual_index = np.argmax(fitness[random_indices])
        winner_index = random_indices[best_individual_index]

        return winner_index

    def tournament_selection_children(self, candidate_indices, population):
        # Generate k unique random indices
        random_indices = np.random.choice(candidate_indices, self.k_tournament, replace=False)
        
        fitness, health_gain, time_game, player_life, enemy_life = self.evaluate(population[random_indices])
        fitness = np.array([self.fitness_function(player_life[i], enemy_life[i], time_game[i]) for i in range(len(player_life))])        
        
        # Select the individuals with the highest fitness values among the randomly chosenpytrel ones
        best_individual_index = random_indices[np.argmax(fitness)]
        
        health_gain = np.array(health_gain[np.argmax(fitness)])
        time_game = np.array(time_game[np.argmax(fitness)])
        
        return best_individual_index, np.max(fitness), health_gain, time_game
    
    def elitism(self, fitness):
        best_indices = np.argsort(fitness)[-self.n_elitism:]
        
        return best_indices
        
    # returns selected individuals and their fitness
    def selection(self, population):        
        #if selection pressure is set to True we want the pressure to increase and thus the number of neural networks to compete to increase
        #eventually the number of individuals in tournament is doubled from start to finish.
        if self.type_of_selection_pressure=="linear":
            self.k_tournament = max(math.ceil(self.current_generation*self.k_tournament_final_linear_increase_factor/self.gens)*self.k_tournament_start, self.k_tournament_start)
        elif self.type_of_selection_pressure=="exponential":
            k_end = self.k_tournament_final_linear_increase_factor*self.k_tournament_start
            exp_rate = 1/self.gens
            self.k_tournament = self.k_tournament_start + math.floor((k_end - self.k_tournament_start) * ((self.current_generation**2)/(self.gens**2)))
        print(f'k is: {self.k_tournament}')
                
        candidate_indices = range(population.shape[0])
        selected_indices = np.zeros(self.n_pop-self.n_elitism)
        fitness = np.zeros(self.n_pop-self.n_elitism)
        health_gain = np.zeros((self.n_pop-self.n_elitism, len(self.env.enemies)))
        time_game = np.zeros((self.n_pop-self.n_elitism, len(self.env.enemies)))
        
        for i in range(self.n_pop-self.n_elitism):
            winner_index, fitness[i], health_gain[i], time_game[i] = self.tournament_selection_children(population.shape[0], population)
            #Remove the selected elite individuals and their fitness values from population and fitness
            selected_indices[i] = winner_index            
            #candidate_indices = np.delete(candidate_indices, np.where(candidate_indices == winner_index))
        
        return population[selected_indices.astype(int)], fitness, health_gain, time_game
    
    def fitness_function(self, player_life, enemy_life, time_game):
        max_health = len(player_life)*100
        
        gains = [playerlife - enemylife for playerlife, enemylife in zip(player_life, enemy_life)]        
        gains_array = np.array(gains)
        weights_gains = np.zeros(len(gains_array))
        weights_times = np.zeros(len(time_game))
        win_count = 0
        #Count less weight to gain if enemy is beaten and dont try to improve time when enemy is beaten
        for i in range(len(gains_array)):
            if gains_array[i] > 0:
                weights_gains[i] = 0.2
                win_count += 1
                time_game[i] = np.max(time_game[time_game != time_game[i]])
            elif gains_array[i] < -100:
                weights_gains[i] = 2
            else:
                weights_gains[i] = 1    
        
        time_score = sum(time_game)/max(time_game)
        #print(time_score)

        #If youre losing the longer time the better, therefore + time
        #print(np.dot(gains_array, weights_gains)/max_health )
        fitness = np.dot(gains_array, weights_gains)/max_health + 0.1*time_score  #+ 0.05 * win_count/len(player_life)        
        return fitness
    
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
        fitness = np.array([self.fitness_function(player_life[i], enemy_life[i], time_game[i]) for i in range(len(player_life))])
        health_gain = np.array([np.mean(i) for i in health_gain])
        time_game = np.array([np.mean(i) for i in time_game])
        
        # Initialize best individual and its fitness
        best_individual_index = np.argmax(fitness)
        best_individual = population[best_individual_index]
        best_fitness = fitness[best_individual_index]

        same_result_count = 0
        reset = False

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
                if self.updated_enemies:
                    fitness, health_gain, time_game, player_life, enemy_life = self.evaluate(population)

                    fitness = np.array([self.fitness_function(player_life[i], enemy_life[i], time_game[i]) for i in range(len(player_life))])
                    health_gain = np.array([np.mean(i) for i in health_gain])
                    time_game = np.array([np.mean(i) for i in time_game])
                    
                    best_individual_index = np.argmax(fitness)
                    best_individual = population[best_individual_index]
                    best_fitness = fitness[best_individual_index]
                
                self.current_generation += 1
                children = []
                for _ in range(self.n_pop*self.lamba_mu_ratio//2):  # Two children per iteration so double the population size out of which we will select the best
                    winner_index1 = self.tournament_selection_parents(population.shape[0], fitness)
                    winner_index2 = self.tournament_selection_parents(population.shape[0], fitness)

                    child1, child2 = self.crossover(population[winner_index1], population[winner_index2], self.number_of_crossovers)
                    
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                    children.extend([child1, child2])                
                
                parents_survivors_indices = self.elitism(fitness)   
                population = np.delete(population, parents_survivors_indices)      
                
                children = np.array(children)
                pop_before_selection = np.concatenate((population, children))
                
                if self.n_elitism > 0:
                    selected_children, fitness_children, health_gain_children, time_game_children = self.selection(pop_before_selection)                        
                    fitness = np.append(fitness[parents_survivors_indices], fitness_children)
                    health_gain = np.append(health_gain[parents_survivors_indices], health_gain_children)
                    time_game = np.append(time_game[parents_survivors_indices], time_game_children)
                    population = np.append(population[parents_survivors_indices], selected_children, axis = 0)   
                else:  
                    population, fitness, health_gain, time_game = self.selection(children) 
                #print(health_gain)
                # Check if any individual has a higher fitness, save that one
                max_fitness_index = np.argmax(fitness)
                if fitness[max_fitness_index] > best_fitness:
                    best_fitness = fitness[max_fitness_index]
                    best_individual = population[max_fitness_index]
                elif fitness[max_fitness_index] == best_fitness:
                    same_result_count += 1
                elif fitness[max_fitness_index] != best_fitness:
                    same_result_count = 0

                if reset:
                    reset = False
                    self.mutation_rate = old_mutation_rate
                    self.crossover_rate = old_crossover_rate
                
                if same_result_count > 3:
                    print('RESET')
                    old_mutation_rate = self.mutation_rate
                    old_crossover_rate = self.crossover_rate                    
                    self.mutation_rate = 1                    
                    self.crossover_rate = 1                    
                    reset = True                
                
                # save to csv
                writer.writerow([gen, np.max(fitness), np.mean(fitness), np.std(fitness),
                                    np.max(health_gain), np.mean(health_gain), np.std(health_gain),
                                    np.min(time_game), np.mean(time_game), np.std(time_game)])
                
                print(f"Generation {gen}, Best Fitness: {np.max(fitness)} and index {np.argmax(fitness)}")
                                
                # Save the best individual's neural network weights
                np.save(os.path.join(self.experiment_dir, "best_individual.npy"), best_individual)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Save the elapsed time
        time_file_path = os.path.join(self.experiment_dir, "training_time_and_effort.txt")
        with open(time_file_path, 'w') as file:
            file.write(f"Training Time: {elapsed_time} seconds\n")
            file.write(f"Tuning effort: {self.counter} \n")

        # this line is only added for the SPO
        self.env.enemies = [1, 2, 3, 4, 5, 6, 7, 8]
        fitnesses, health_gains, times, player_lifes, enemy_lifes = self.simulation(best_individual[:-1])
        best_fitness = self.fitness_function(player_lifes, enemy_lifes, times)
        
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

            # Run the simulation with the best individual[
            fitness, health_gain, time, player_life, enemy_life = self.simulation(best_individual[:-1])
            fitness = self.fitness_function(player_life, enemy_life, time)

            print(f'fitness is: {fitness}')
            print(f'health gain is: {health_gain}')
            print(f'total health gain is: {np.sum(health_gain)}')
            print(f'player life is: {player_life}')
            print(f'enemy life is: {enemy_life}')
            print(f'time is: {time}')

        else:
            print("No best individual found!")


def run_evoman(experiment_name, enemies, population_size, generations, mutation_rate, crossover_rate, mode, 
               n_hidden_neurons, headless, dom_l, dom_u, speed, number_of_crossovers, n_elitism, k_tournament, type_of_selection_pressure, 
               k_tournament_final_linear_increase_factor, alpha, enemy_threshold, lamba_mu_ratio):
        evoman = EvoMan(experiment_name, enemies, population_size, generations, mutation_rate, crossover_rate, 
                        mode, n_hidden_neurons, headless, dom_l, dom_u, speed, number_of_crossovers, n_elitism, k_tournament, 
                        type_of_selection_pressure, k_tournament_final_linear_increase_factor, alpha, enemy_threshold, lamba_mu_ratio)
        
        # Log the command
        if mode == "train":
            log_file_path = os.path.join(evoman.experiment_dir, "commands_log.txt")
            with open(log_file_path, "a") as f:
                f.write(' '.join(sys.argv) + '\n')
        
        fitness = evoman.run()
        
        return fitness
        

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Evolutionary Algorithm for EvoMan")
        
        parser.add_argument("--experiment_name", type=str, default="experiment", help="Name of the experiment")
        parser.add_argument("--enemies", type=int, nargs='+',default=[1, 2, 3, 4, 5, 6, 7, 8], help="Enemies numbers")
        parser.add_argument("--npop", type=int, default=100, help="Size of the population")
        parser.add_argument("--gens", type=int, default=30, help="Number of generations")
        parser.add_argument("--mutation_rate", type=float, default=0.2, help="Mutation rate")
        parser.add_argument("--crossover_rate", type=float, default=0.7, help="Crossover rate")
        parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")
        parser.add_argument("--n_hidden_neurons", type=int, default=10, help="Number of hidden neurons")
        parser.add_argument("--headless", action="store_true", help="Run in headless mode")
        parser.add_argument("--dom_l", type=float, default=-1, help="Lower bound for initialization and mutation")
        parser.add_argument("--dom_u", type=float, default=1, help="Upper bound for initialization and mutation")
        parser.add_argument("--speed", type=str, default="fastest", help="Speed: fastest or normal")
        parser.add_argument("--number_of_crossovers", type=int, default=5, help="Number of crossovers")
        parser.add_argument("--n_elitism", type=int, default=2, help="Number of best individuals from population that are always selected for the next generation.")
        parser.add_argument("--k_tournament", type=int, default= 2, help="The amount of individuals to do a tournament with for selection, the more the higher the selection pressure")
        parser.add_argument("--type_of_selection_pressure", type=str, default="exponential", help="if set to linear the selection pressure will linearly increase over time from k_tournament till k_tournament_final_linear_increase_factor*k_tournament, if set to exponential the selection pressure will increase exponentially from k_tournament till 2*k_tournament, if set to anything else the selection pressure will stay the same")
        parser.add_argument("--k_tournament_final_linear_increase_factor", type=int, default= 4, help="The factor with which k_tournament should linearly increase (if type_of_selection_pressure = True), if the value is 4 the last quarter of generations have tournaments of size k_tournament*4")
        parser.add_argument("--alpha", type=float, default=0.5, help="Weight for enemy damage")
        parser.add_argument("--enemy_threshold", type=int, default=15, help="The threshold health gain from which an enemy will be swapped with another enemy to train.")
        parser.add_argument("--lamba_mu_ratio", type=int, default=3, help="Ratio between lamda and mu")

        args = parser.parse_args()

        run_evoman(args.experiment_name, args.enemies, args.npop, args.gens, args.mutation_rate, args.crossover_rate,
               args.mode, args.n_hidden_neurons, args.headless, args.dom_l, args.dom_u, args.speed, args.number_of_crossovers,
               args.n_elitism, args.k_tournament, args.type_of_selection_pressure, args.k_tournament_final_linear_increase_factor,
               args.alpha, args.enemy_threshold, args.lamba_mu_ratio)