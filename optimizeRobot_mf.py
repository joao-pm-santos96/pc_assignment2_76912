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
import logging
logger = logging.getLogger(__name__)

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
        angles = [30.0, 90.0, -30.0, -90.0]

        rob=MyRob(rob_name, pos, angles, host, 
            base_speed=solution[0], 
            P=solution[1], 
            I=solution[2], 
            D=solution[3],
            in_eval=True)
        
        rob.run()

        return rob.measures.score

    @staticmethod
    def fitness_wrapper(idx, solution):
        return PooledGA.fitness_func(solution, idx+1)

    @staticmethod
    def start_sim():

        logger.info('Starting SIM...')
        explorer = {'x':430, 'y':300}
        region = (25, 25, (140-25), (140-25))
        confidence = 0.95
        sleep = 0.25

        while pyautogui.locateOnScreen('running.png', region=region, confidence=confidence) is None:
            pyautogui.click(x=explorer['x'], y=explorer['y'], clicks=1, button='left')
            pyautogui.hotkey('ctrl', 's')
            time.sleep(sleep)

        logger.info('Done')
            
        # pos = None
        # while pos is None:
        #     pos = pyautogui.locateCenterOnScreen('start.png', region=(0,0, 200, 200), confidence=0.75)

        # while pos is not None:
        #     pyautogui.click(pos[0],pos[1])
        #     logger.debug('Click START')
        #     time.sleep(0.1)
        #     pos = pyautogui.locateCenterOnScreen('start.png', region=(0,0, 200, 200), confidence=0.5)

        # logger.info('Started SIM')


    @staticmethod
    def reset_sim():
        explorer = {'x':430, 'y':300}

        # reset simulation
        pyautogui.click(x=explorer['x'], y=explorer['y'], clicks=1, button='left')
        pyautogui.hotkey('ctrl', 'r')
        logger.debug('Reset SIM')

    @staticmethod
    def command_sim(command):
        explorer = {'x':430, 'y':300}

        # reset simulation
        pyautogui.click(x=explorer['x'], y=explorer['y'], clicks=1, button='left')

        if command == 'reset':
            pyautogui.hotkey('ctrl', 'r')
            logger.debug('Reset SIM')
        elif command == 'start':
            pyautogui.hotkey('ctrl', 's')
            logger.debug('Start SIM')

    @staticmethod
    def on_fitness(ga, solutions_fitness):
        PooledGA.reset_sim() # must!        

    @staticmethod
    def on_generation(ga):
        max_fitness = np.amax(ga.last_generation_fitness)
        max_index = np.argmax(ga.last_generation_fitness)

        logger.info(f'Generation: {ga.generations_completed} of {ga.num_generations}')
        logger.info(f'Best fitness: {max_fitness}')
        logger.info(f'Best solution: {ga.population[max_index]}')
        
    @staticmethod
    def on_start(ga):
        PooledGA.reset_sim()  

    def cal_pop_fitness(self):

        # t = Timer(0.1, PooledGA.start_sim)
        # t.start()
        
        with Pool(processes=self.sol_per_pop) as pool:
            # pop_fitness = pool.starmap(PooledGA.fitness_wrapper, list(enumerate(self.population)))   
            pop_fitness = pool.starmap_async(PooledGA.fitness_wrapper, list(enumerate(self.population)))  

            # start sim
            time.sleep(1)
            PooledGA.command_sim('start')

            # wait
            pool.close()
            pool.join() # TODO check RAM

            print('now done')


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
    formatter = logging.Formatter(' %(asctime)s [%(levelname)s]: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

"""
MAIN
"""
if __name__ == '__main__':

    configLogger()

    # gene_space = [{'low': 0,'high': 90}, {'low': 91,'high': 180}, None, None, None, None] 
    # gene_type = [int, int, [float, 3], [float, 3], [float, 3], [float, 3]]

    num_genes = 4
    num_generations = 100
    sol_per_pop = 50
    num_parents_mating = 10

    # gene_space = None
    gene_space = None

    gene_type = [float, 5]

    gene_init_val = 1.0
    random_mutation_val = 0.5

    pyautogui.PAUSE = 0.1

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
                    mutation_probability=0.5,
                    parent_selection_type="sus",
                    crossover_type="uniform",
                    mutation_type="random",
                    allow_duplicate_genes=False,
                    save_best_solutions=False)


    ga.compute()
