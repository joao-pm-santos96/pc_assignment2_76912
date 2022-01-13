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
from multiprocessing import Pool
# from threading import Timer
from datetime import datetime
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
        base_speed, P, I, D, angle0, angle1, weight = solution
        angles = [angle0, angle1, -angle0, -angle1]

        rob=MyRob(rob_name, pos, angles, host, 
            base_speed=base_speed,
            P=P,
            I=I,
            D=D,
            weight=weight,
            in_eval=True)
        
        rob.run()

        return rob.measures.score

    @staticmethod
    def fitness_wrapper(idx, solution):
        return PooledGA.fitness_func(solution, idx+1)

    @staticmethod
    def command_sim(command):
        explorer = {'x':430, 'y':300}

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
        max_fitness = np.max(ga.last_generation_fitness)

        logger.info(f'Generation: {ga.generations_completed} of {ga.num_generations}')
        for idx in np.where(ga.last_generation_fitness == max_fitness):
            logger.info(f'Best solution({idx}): {ga.population[idx]}')
        logger.info(f'Best fitness: {max_fitness}')
        logger.info('='*20)
        
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

    num_genes = 7
    num_generations = 1000
    sol_per_pop = 100
    num_parents_mating = 15

    gene_space = [{'low': 0,'high': 1}, None, None, None, {'low': 0,'high': 180}, {'low': 0,'high': 180}, {'low': 0,'high': 1}] # TODO test this one
    gene_type = [[float, 6], [float, 6], [float, 6], [float, 6], int, int, [float, 6]]

    gene_init_val = 1.0
    random_mutation_val = 0.5    

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
                       mutation_probability=0.4,
                       parent_selection_type="sus",
                       crossover_type="uniform",
                       mutation_type="random",
                       allow_duplicate_genes=False,
                       save_best_solutions=False,
                       stop_criteria="saturate_150")

    ga.compute()
