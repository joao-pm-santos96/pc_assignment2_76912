#!/usr/bin/env python3
import sys
from croblink import *
from math import *
import xml.etree.ElementTree as ET

"""
JS Imports
"""
import numpy as np
from pygad import nn

CELLROWS=7
CELLCOLS=14

class MyRob(CRobLinkAngs):
    def __init__(self, 
                rob_name, 
                rob_id, 
                angles, 
                host,  
                in_eval= False):

        CRobLinkAngs.__init__(self, rob_name, rob_id, angles, host)

        self.in_eval = in_eval
        self.nn = None

    # In this map the center of cell (i,j), (i in 0..6, j in 0..13) is mapped to labMap[i*2][j*2].
    # to know if there is a wall on top of cell(i,j) (i in 0..5), check if the value of labMap[i*2+1][j*2] is space or not
    def setMap(self, labMap):
        self.labMap = labMap

    def printMap(self):
        for l in reversed(self.labMap):
            print(''.join([str(l) for l in l]))

    def run(self):
        if self.status != 0:
            print("Connection refused or error")

            if self.in_eval:
                self.measures = CMeasures()
                self.measures.score = 0
                return # test this (error intended to not forget ;) )
            else:
                quit()

        state = 'stop'
        stopped_state = 'run'

        # localization
        last_ground = -1
        grounds = []

        last_pos = None
        self.distance = 0

        while True:
            self.readSensors()

            if self.in_eval and last_pos is not None:
                pos = (self.measures.x, self.measures.y)
                self.distance += np.linalg.norm(np.array(pos) - np.array(last_pos))
                last_pos = pos

            if self.in_eval and last_pos is None:
                last_pos = (self.measures.x, self.measures.y)

            # localization
            if (self.measures.ground != last_ground) and self.measures.ground != -1:
                grounds.append(self.measures.ground)
                last_ground = self.measures.ground

            # stop conditions

            # collision
            if self.in_eval and self.measures.collision:
                return

            # taking too long
            if self.in_eval and not self.measures.start and not self.measures.stop:
                return

            # PID
            # delta1 = (self.measures.irSensor[2] - self.measures.irSensor[0])
            # delta2 = (self.measures.irSensor[3] - self.measures.irSensor[1])
            # self.pid.update(self.weight*delta1 + (1-self.weight)*delta2)
            # val = np.sum(np.multiply(self.weights, self.measures.irSensor))
            # self.pid.update(val)

            if self.measures.endLed:
                print(self.rob_name + " exiting")
                quit()

            if state == 'stop' and self.measures.start:
                state = stopped_state

            if state != 'stop' and self.measures.stop:
                stopped_state = state
                state = 'stop'

            if state == 'run':
                if self.measures.visitingLed==True:
                    state='wait'
                if self.measures.ground==0:
                    self.setVisitingLed(True);
                self.wander()
            elif state=='wait':
                self.setReturningLed(True)
                if self.measures.visitingLed==True:
                    self.setVisitingLed(False)
                if self.measures.returningLed==True:
                    state='return'
                self.driveMotors(0.0,0.0)
            elif state=='return':
                if self.measures.visitingLed==True:
                    self.setVisitingLed(False)
                if self.measures.returningLed==True:
                    self.setReturningLed(False)
                self.wander()


    def wander(self):

        predictions = nn.predict(last_layer=self.nn,
                               data_inputs=np.array([self.measures.irSensor]),
                               problem_type='regression')

        self.driveMotors(predictions[0][0], predictions[0][1])

    """
    JS METHODS
    """
    def checkTravelDir(self, last_two):
        return tuple(last_two) in [(0,1), (1,2), (2,0)] if len(last_two) > 1 else True

    def checkLapCompleted(self, last_four):
        return tuple(last_four) == (0, 1, 2, 0) if len(last_four) == 4 else False

    def createNetwork(self,
                    num_neurons_input, 
                    num_neurons_output, 
                    num_neurons_hidden_layers=[], 
                    output_activation="softmax", 
                    hidden_activations="relu"):

        # Creating the input layer as an instance of the nn.InputLayer class.
        input_layer = nn.InputLayer(num_neurons_input)

        if type(hidden_activations) not in [list,tuple]:
            hidden_activations = [hidden_activations]*len(num_neurons_hidden_layers)

        if len(num_neurons_hidden_layers) > 0:
            # If there are hidden layers, then the first hidden layer is connected to the input layer.
            hidden_layer = nn.DenseLayer(num_neurons=num_neurons_hidden_layers.pop(0), 
                                        previous_layer=input_layer, 
                                        activation_function=hidden_activations.pop(0))
            # For the other hidden layers, each hidden layer is connected to its preceding hidden layer.
            for hidden_layer_idx in range(len(num_neurons_hidden_layers)):
                hidden_layer = nn.DenseLayer(num_neurons=num_neurons_hidden_layers.pop(0), 
                                            previous_layer=hidden_layer, 
                                            activation_function=hidden_activations.pop(0))

            # The last hidden layer is connected to the output layer.
            # The output layer is created as an instance of the nn.DenseLayer class.
            output_layer = nn.DenseLayer(num_neurons=num_neurons_output, 
                                        previous_layer=hidden_layer, 
                                        activation_function=output_activation)

        # If there are no hidden layers, then the output layer is connected directly to the input layer.
        elif len(num_neurons_hidden_layers) == 0:
            # The output layer is created as an instance of the nn.DenseLayer class.
            output_layer = nn.DenseLayer(num_neurons=num_neurons_output, 
                                        previous_layer=input_layer,
                                        activation_function=output_activation)

        # Returning the reference to the last layer in the network architecture which is the output layer. Based on such reference, all network layer can be fetched.
        self.nn = output_layer

    def updateNNWeights(self, weights):
        weights_matrix = nn.layers_weights_as_matrix(self.nn, weights)
        nn.update_layers_trained_weights(last_layer=self.nn,
                                    final_weights=weights_matrix)

class Map():
    def __init__(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()

        self.labMap = [[' '] * (CELLCOLS*2-1) for i in range(CELLROWS*2-1) ]
        i=1
        for child in root.iter('Row'):
           line=child.attrib['Pattern']
           row =int(child.attrib['Pos'])
           if row % 2 == 0:  # this line defines vertical lines
               for c in range(len(line)):
                   if (c+1) % 3 == 0:
                       if line[c] == '|':
                           self.labMap[row][(c+1)//3*2-1]='|'
                       else:
                           None
           else:  # this line defines horizontal lines
               for c in range(len(line)):
                   if c % 3 == 0:
                       if line[c] == '-':
                           self.labMap[row][c//3*2]='-'
                       else:
                           None

           i=i+1


rob_name = "pClient1"
host = "localhost"
pos = 1
mapc = None

for i in range(1, len(sys.argv),2):
    if (sys.argv[i] == "--host" or sys.argv[i] == "-h") and i != len(sys.argv) - 1:
        host = sys.argv[i + 1]
    elif (sys.argv[i] == "--pos" or sys.argv[i] == "-p") and i != len(sys.argv) - 1:
        pos = int(sys.argv[i + 1])
    elif (sys.argv[i] == "--robname" or sys.argv[i] == "-r") and i != len(sys.argv) - 1:
        rob_name = sys.argv[i + 1]
    elif (sys.argv[i] == "--map" or sys.argv[i] == "-m") and i != len(sys.argv) - 1:
        mapc = Map(sys.argv[i + 1])
    else:
        print("Unkown argument", sys.argv[i])
        quit()

if __name__ == '__main__':

    # setup agent
    solution = [4.400000000000000000e+01,
5.200000000000000000e+01,
-2.226930000000000021e-01,
3.276270000000000016e-01,
-2.641814000000000107e+00,
-4.662047000000000274e+00,
-1.752316999999999902e+00,
1.984593999999999969e+00,
2.062990999999999797e+00,
-8.743859999999999966e-01,
6.470614000000000310e+00,
3.659292999999999907e+00,
-7.286169999999999591e-01,
2.747415999999999858e+00,
3.232791999999999888e+00,
3.576950000000000074e+00,
-2.323364999999999903e+00,
4.809910000000000352e+00,
2.480538999999999827e+00,
-3.205426000000000109e+00,
-5.913583000000000034e+00,
3.358883000000000063e+00,
1.635742000000000029e+00,
6.748383999999999716e+00,
1.370146000000000086e+00,
1.368260000000000032e-01,
-4.067439999999999944e-01,
1.855008999999999908e+00,
5.790213999999999750e+00,
-1.535288999999999904e+00,
9.511290000000000022e-01,
4.225881000000000220e+00,
-4.575586000000000375e+00,
-4.223439999999999972e-01,
5.122027000000000108e+00,
-4.804202000000000083e+00]

    
    angles = [solution[0], solution[1], -solution[0], -solution[1]]
    weights = solution[2:]

    rob=MyRob(rob_name, pos, angles, host, 
        in_eval=True)

    if mapc != None:
        rob.setMap(mapc.labMap)
        rob.printMap()    

    rob.createNetwork(4, 2, [4, 3], output_activation='None') 
    rob.updateNNWeights(weights)
    rob.run()



