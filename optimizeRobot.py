#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
from signal import raise_signal
from mainRob import MyRob
from multiprocessing import Pool
from datetime import datetime
from tabulate import tabulate

import pygad
import pyautogui
import time
# from threading import Timer
import numpy as np
import logging
logger = logging.getLogger(__name__)

"""
METADATA
"""
__author__ = 'Joao Santos'
__copyright__ = 'Copyright December2021'
__credits__ = ['Joao Santos']
__version__ = '1.0.0'
__maintainer__ = 'Joao Santos'
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
        base_speed, P, I, D, windup, alpha0, alpha1, w0, w1, setpoint, Ksr = solution
        angles = [alpha0, alpha1, -alpha0, -alpha1]
        weights = [w0, w1, -w0, -w1, Ksr]

        rob=MyRob(rob_name, pos, angles, host, 
            base_speed=base_speed,
            P=P,
            I=I,
            D=D,
            set_point=setpoint,
            windup=windup,
            weights=weights,
            in_eval=True)
        
        rob.run()

        return rob.measures.score

    @staticmethod
    def fitness_wrapper(idx, solution):
        return PooledGA.fitness_func(solution, idx+1)

    @staticmethod
    def command_sim(command):
        explorer = {'x':100, 'y':300}

        # reset simulation
        pyautogui.click(x=explorer['x'], y=explorer['y'], clicks=1, button='left')

        if command == 'reset':
            pyautogui.hotkey('ctrl', 'r')
            logger.debug('Reset Simulator')
        elif command == 'start':
            pyautogui.hotkey('ctrl', 's')
            logger.debug('Start Simulator')

    @staticmethod
    def on_fitness(ga, solutions_fitness):
        PooledGA.command_sim('reset') # must!        

    @staticmethod
    def on_generation(ga):
        headers = ['base_speed', 'P', 'I', 'D', 'windup', 'alpha0', 'alpha1', 'w0', 'w1', 'SP', 'Ksr']
        sol, fit, idx = ga.best_solution(pop_fitness=ga.last_generation_fitness)
        
        logger.info(f'Generation: {ga.generations_completed} of {ga.num_generations}')
        logger.info(f'Best fitness {fit} [{idx}]')
        logger.info(f'Mean fitness: {np.mean(ga.last_generation_fitness)}')
        
        print()
        print(tabulate([sol], headers=headers))
        print('='*50)
        
    @staticmethod
    def on_start(ga):
        PooledGA.command_sim('reset') 

    @staticmethod
    def on_stop(ga, last_population_fitness):
        PooledGA.command_sim('reset') 

    def cal_pop_fitness(self):
        
        with Pool(processes=self.sol_per_pop) as pool:
            
            pop_fitness = pool.starmap_async(PooledGA.fitness_wrapper, list(enumerate(self.population)))  

            # start sim
            time.sleep(1)
            PooledGA.command_sim('start')

            # wait
            pool.close()
            pool.join() 

        return np.array(pop_fitness.get())

    def compute(self):

        self.run()

        solution, solution_fitness, solution_idx = self.best_solution()
        logger.info("Parameters of the best solution : {solution}".format(solution=solution))
        logger.info("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

        name_append =f'{round(solution_fitness, 3)}_{datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}'

        self.save(filename=f'outputs/genetic_{name_append}')

        filename = f'outputs/results_{name_append}.txt' 
        np.savetxt(filename, solution, delimiter=',')

        self.plot_fitness(save_dir=f'outputs/graph_{name_append}.png')
   
"""
FUNCTIONS DEFINITIONS
"""
def configLogger():
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

"""
MAIN
"""
if __name__ == '__main__':

    configLogger()
    pyautogui.PAUSE = 0.1

    gene_space = [{'low': 0,'high': 1}, # linear speed (must be positive, power ratio)
                    None, # P
                    None, # I
                    None, # D
                    {'low': 0,'high': 50}, # windup (must be positive)
                    {'low': 0,'high': 90}, # alpha0
                    {'low': 0,'high': 90}, # alpha1
                    None, # weight0
                    None, # weight1
                    [0], # set-point
                    [0], # Ksr (must be positive)
                    ] 

    gene_type = [[float, 6], # linear speed
                [float, 6], # P
                [float, 6], # I
                [float, 6], # D
                [float, 6], # windup
                int, # alpha0
                int, # alpha1
                [float, 6], # weight0
                [float, 6], # weight1
                [float, 6], # set-point
                [float, 6] # Ksr
                ]

    if len(gene_space) != len(gene_type):
        raise Exception('== GENE SPACE and GENE TYPE have different lengths ==')

    num_genes = len(gene_space)
    num_generations = 500
    sol_per_pop = 100
    num_parents_mating = 25

    gene_init_val = 1.0
    random_mutation_val = 2.0    

    ga = PooledGA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_type=gene_type,
                       gene_space=gene_space,
                       init_range_high=gene_init_val,
                       init_range_low=-gene_init_val, 
                       random_mutation_min_val = -random_mutation_val,
                       random_mutation_max_val = random_mutation_val,
                       fitness_func=PooledGA.fitness_func,
                       on_generation=PooledGA.on_generation,
                       on_fitness=PooledGA.on_fitness, 
                       on_stop=PooledGA.on_stop,
                       mutation_probability=0.15,
                       parent_selection_type="sus",
                       crossover_type="uniform",
                       mutation_type="random",
                       allow_duplicate_genes=True,
                       save_best_solutions=False)

    ga.compute()
