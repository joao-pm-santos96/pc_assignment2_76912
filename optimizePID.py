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
import json
import time

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

    # reset simulation
    pyautogui.hotkey('ctrl', 'r')

    # setup agent
    angles = [solution[0], solution[1], -1*solution[0], -1*solution[1]]
    
    rob=MyRob(rob_name, pos, angles, host, 
    base_speed=solution[2], 
    P=solution[3], 
    I=solution[4], 
    D=solution[5],
    in_eval=True)
    
    rob.run()

    # start simulation
    pyautogui.hotkey('ctrl', 's')

    return ?

def on_generation(ga):
    print("Generation", ga.generations_completed)
    # print(ga.population)

"""
MAIN
"""
if __name__ == '__main__':
    
    num_generations = 50
    num_parents_mating = 4

    initial_population = [45, 90, 0.1, 0, 0, 0]
    gene_space = [range(0, 90), range(90, 181), None, None, None, None] 
    gene_type = [int, int, [float, 3]*4]

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       initial_population=initial_population,
                       gene_type=gene_type,
                       gene_space=gene_space,                       
                       parent_selection_type="sss",
                       keep_parents=1,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=10,
                       on_generation=on_generation,
                       allow_duplicate_genes=False)

    ga_instance.run()

    ga_instance.save(filename='genetic')

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    data = {'solution': solution,
        'fitness': solution_fitness,
        'index': solution_idx}

    with open(f'results_{solution_fitness}_{time.time()}.txt', 'w') as outfile:
        json.dump(data, outfile)

    ga_instance.plot_fitness()


    