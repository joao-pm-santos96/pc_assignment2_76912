#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
from mainRob import MyRob

import pygad
import pyautogui
import time
from multiprocessing import Pool, current_process
from threading import Timer
from datetime import datetime
import numpy as np

"""
METADATA
"""
__author__ = 'Joao Santos'
__copyright__ = 'Copyright December2021, Motofil, S.A.'
__credits__ = ['Joao Santos']
__version__ = '1.0.0'
__maintainer__ = 'Joao Santos'
__email__ = 'jpsantos@motofil.com'
__status__ = 'Production'
# __license__ = 'GPL'

"""
TODO
"""

"""
CLASS DEFINITIONS
"""
class PooledGA(pygad.GA):

    @staticmethod
    def fitness_func(solution, index):
        rob_name = "PID"
        host = "localhost"
        pos = int(index)

        # setup agent
        angles = [0.0, 45.0, -45.0, 180.0]

        rob=MyRob(rob_name, pos, angles, host, 
            base_speed=solution[3], 
            P=solution[0], 
            I=solution[1], 
            D=solution[2],
            in_eval=True)
        
        rob.run()

        return rob.distance

    @staticmethod
    def fitness_wrapper(idx, solution):
        return PooledGA.fitness_func(solution, idx+1)

    @staticmethod
    def start_sim():
        explorer = {'x':200, 'y':845}

        # start simulation
        pyautogui.click(x=explorer['x'], y=explorer['y'], clicks=1, button='left')
        pyautogui.hotkey('ctrl', 's')

    @staticmethod
    def reset_sim():
        explorer = {'x':200, 'y':845}

        # reset simulation
        pyautogui.click(x=explorer['x'], y=explorer['y'], clicks=1, button='left')
        pyautogui.hotkey('ctrl', 'r')

    @staticmethod
    def on_fitness(ga, solutions_fitness):
        PooledGA.reset_sim() # must!        

    @staticmethod
    def on_generation(ga):
        print(f'Generation: {ga.generations_completed}')
        print(f'Best solution: {ga.best_solutions[-1]}')
        print(f'Best fitness: {ga.best_solutions_fitness[-1]}')
        
    @staticmethod
    def on_start(ga):
        PooledGA.reset_sim()  

    def cal_pop_fitness(self):

        t = Timer(0.1, PooledGA.start_sim)
        t.start()
        
        with Pool(processes=self.sol_per_pop) as pool:
            pop_fitness = pool.starmap(PooledGA.fitness_wrapper, list(enumerate(self.population)))   

        return np.array(pop_fitness)

    def compute(self):

        self.run()

        solution, solution_fitness, solution_idx = self.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

        name_append =f'{round(solution_fitness, 3)}_{datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}'

        self.save(filename=f'outputs/genetic_{name_append}')

        filename = f'outputs/results_{name_append}.txt' 
        np.savetxt(filename, solution, delimiter=',')

        self.plot_fitness(save_dir=f'outputs/graph_{name_append}.png')
   
"""
FUNCTIONS DEFINITIONS
"""

"""
MAIN
"""
if __name__ == '__main__':


    # num_genes = 6

    # num_generations = 1000
    # sol_per_pop = 25
    # num_parents_mating = int(sol_per_pop * 0.1)

    # gene_space = [{'low': 0,'high': 90}, {'low': 91,'high': 180}, None, None, None, None] 
    # gene_type = [int, int, [float, 3], [float, 3], [float, 3], [float, 3]]


    num_genes = 4

    num_generations = 1000
    sol_per_pop = 50
    num_parents_mating = int(sol_per_pop * 0.5) if sol_per_pop >= 10 else 1

    gene_space = None
    gene_type = float

    gene_init_val = 1.0
    random_mutation_val = 0.5

    instance = PooledGA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=PooledGA.fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_type=gene_type,
                       gene_space=gene_space,
                       parent_selection_type="sus",
                       keep_parents=1,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=10,
                       init_range_high=gene_init_val,
                       init_range_low=-gene_init_val, 
                       random_mutation_min_val = -random_mutation_val,
                       random_mutation_max_val = random_mutation_val,
                       on_generation=PooledGA.on_generation,
                       on_fitness=PooledGA.on_fitness,  
                       allow_duplicate_genes=True,
                       save_best_solutions=True)


    instance.compute()





