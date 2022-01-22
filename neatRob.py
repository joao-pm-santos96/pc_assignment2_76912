#!/usr/bin/env python3
import sys
from croblink import *
from math import *
import xml.etree.ElementTree as ET

"""
JS Imports
"""
import numpy as np
import neat

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
                return 
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

        output = self.nn.activate(self.measures.irSensor)
        self.driveMotors(output[0], output[1])

    """
    JS METHODS
    """
    def checkTravelDir(self, last_two):
        return tuple(last_two) in [(0,1), (1,2), (2,0)] if len(last_two) > 1 else True

    def checkLapCompleted(self, last_four):
        return tuple(last_four) == (0, 1, 2, 0) if len(last_four) == 4 else False

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

   pass