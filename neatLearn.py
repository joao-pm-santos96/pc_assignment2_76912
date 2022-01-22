#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import os
import neat
import pyautogui
import time

from neatRob import MyRob
from multiprocessing import Pool

import logging
logger = logging.getLogger(__name__)
"""
METADATA
"""
__author__ = 'Joao Santos'
__copyright__ = 'Copyright January2022'
__credits__ = ['Joao Santos']
__version__ = '1.0.0'
__maintainer__ = 'Joao Santos'
__email__ = 'joao.pm.santos96@gmail.com'
__status__ = 'Production'
# __license__ = 'GPL'

"""
TODO
"""

"""
CLASS DEFINITIONS
"""

"""
FUNCTIONS DEFINITIONS
"""
def eval_genome(pos, genome, config):

    host = "localhost"
    pos = int(pos+1)
    rob_name = "neat_" + str(pos)
    
    angles = [10, -10, 50, -50] # TODO

    net = neat.nn.FeedForwardNetwork.create(genome, config)      

    rob = MyRob(rob_name, pos, angles, host, in_eval = True)
    rob.nn = net

    rob.run()

    return rob.measures.score

def single_eval(genomes, config):

    for genome_idx, genome in genomes:
        command_sim('reset')

        with Pool(1) as pool:
            fitness = pool.starmap_async(eval_genome, [(0, genome, config)])

            time.sleep(1)
            command_sim('viewer')
            command_sim('start')

            # wait
            pool.close()
            pool.join()

        genome.fitness = fitness.get()[0]

def pool_eval(genomes, config):

    command_sim('reset')
    time.sleep(0.1)

    with Pool(len(genomes)) as pool:
        pop_fitness = pool.starmap_async(eval_genome, [(idx, genome[1], config) for idx, genome in enumerate(genomes)])
        
        # start sim
        time.sleep(0.1)
        command_sim('start')

        # wait
        pool.close()
        pool.join() 

    fitness = pop_fitness.get()
    for genome_id, genome in genomes:
        genome.fitness = fitness[(genome_id - 1)%len(genomes)]

def run(config_file):

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run 
    winner = p.run(pool_eval, 1000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))










def command_sim(command):

    simulator = {'x':430, 'y':150}
    viewer = {'x':430, 'y':750}

    if command == 'reset':
        pyautogui.click(x=simulator['x'], y=simulator['y'], clicks=1, button='left')
        pyautogui.hotkey('ctrl', 'r')
        logger.debug('Reset Simulator')
    elif command == 'start':
        pyautogui.click(x=simulator['x'], y=simulator['y'], clicks=1, button='left')
        pyautogui.hotkey('ctrl', 's')
        logger.debug('Start Simulator')
    elif command == 'viewer':
        pyautogui.click(x=viewer['x'], y=viewer['y'], clicks=1, button='left')
        pyautogui.hotkey('ctrl', 'c')
        logger.debug('Start Viewer')

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
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_neat')
    run(config_path)