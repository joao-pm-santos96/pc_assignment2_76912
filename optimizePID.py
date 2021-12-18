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


"""
FUNCTIONS DEFINITIONS
"""
def fitness_func(solution, solution_idx):

    rob_name = "PID"
    host = "localhost"
    pos = 1

    print(f'Individual {solution_idx}: {solution}')

    explorer = {'x':1415, 'y':845}
    viewer = {'x':1415, 'y':200}

    # reset simulation
    pyautogui.click(x=explorer['x'], y=explorer['y'], clicks=1, button='left')
    pyautogui.hotkey('ctrl', 'r')

    # setup agent
    # angles = [solution[0], solution[1], -1*solution[0], -1*solution[1]]
    angles = [45, 90, -45, -90]
    
    rob=MyRob(rob_name, pos, angles, host, 
    base_speed=0.1, 
    P=solution[0], 
    I=solution[1], 
    D=solution[2],
    in_eval=True)

    # connect viewer
    pyautogui.click(x=viewer['x'], y=viewer['y'], clicks=1, button='left')
    pyautogui.hotkey('ctrl', 'c')

    # start simulation
    pyautogui.click(x=explorer['x'], y=explorer['y'], clicks=1, button='left')
    pyautogui.hotkey('ctrl', 's')

    rob.run()

    # normalization values

    a = 600.0
    b = 400.0
    # print(f'Time: {rob.measures.time}')
    # print(f'Score: {rob.measures.score}')

    # return (1/(rob.measures.time / a) + rob.measures.score / b) * 100.0
    return rob.measures.score


def on_generation(ga):
    print(f'Generation: {ga.generations_completed}')
    print(f'Best solution: {ga.best_solutions[-1]}')
    print(f'Best fitness: {ga.best_solutions_fitness[-1]}')

    # print(ga.population)

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


    num_genes = 3

    num_generations = 100
    sol_per_pop = 25
    num_parents_mating = int(sol_per_pop * 0.1)

    gene_space = None
    gene_type = [float, 3]

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_type=gene_type,
                       gene_space=gene_space,                       
                       parent_selection_type="sus",
                       keep_parents=1,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=10,
                       random_mutation_min_val = -1.0,
                       random_mutation_max_val = 1.0,
                       on_generation=on_generation,
                       allow_duplicate_genes=False,
                       save_best_solutions=True)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    name_append =f'{round(solution_fitness, 3)}_{datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}'

    ga_instance.save(filename=f'outputs/genetic_{name_append}')

    filename = f'outputs/results_{name_append}.txt' 
    np.savetxt(filename, solution, delimiter=',')

    ga_instance.plot_fitness(save_dir=f'outputs/graph_{name_append}.png')
    

    