import sys
from croblink import *
from math import *
import xml.etree.ElementTree as ET

"""
JS Imports
"""
import PID

CELLROWS=7
CELLCOLS=14

class MyRob(CRobLinkAngs):
    def __init__(self, rob_name, rob_id, angles, host, base_speed = 0.1, P=0, I=0, D=0, in_eval= False):
        CRobLinkAngs.__init__(self, rob_name, rob_id, angles, host)

        self.base_speed = base_speed
        self.P = P
        self.I = I
        self.D = D
        self.set_point = 0.0
        self.sample_time = 0.1
        self.in_eval = in_eval

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
            quit()

        state = 'stop'
        stopped_state = 'run' 

        # initialize PID
        self.pid = PID.PID(self.P, self.I, self.D)
        self.pid.SetPoint = self.set_point
        self.pid.setSampleTime(self.sample_time)

        # localization
        last_ground = -1
        grounds = []

        while True:
            self.readSensors()

            # localization
            if (self.measures.ground != last_ground) and self.measures.ground != -1:
                grounds.append(self.measures.ground)
                print(grounds)
                self.measures.ground = last_ground
            
            # stop conditions

            # reached end
            if self.in_eval and self.checkLapCompleted(grounds[-4:]):
                print('Reached end')
                quit()

            # collision
            if self.in_eval and self.measures.collision:
                print('Collided')
                # TODO penalty?
                quit()

            # taking too long
            if self.in_eval and self.measures.time > 5000:
                print('Took too long')
                quit()

            # PID
            delta = (self.measures.irSensor[0] - self.measures.irSensor[2]) + (self.measures.irSensor[1] - self.measures.irSensor[3])
            self.pid.update(delta)

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
        # center_id = 0
        # left_id = 1
        # right_id = 2
        # back_id = 3

        self.driveMotors(self.base_speed - self.pid.output, 
                         self.base_speed + self.pid.output)

        # if    self.measures.irSensor[center_id] > 5.0\
        #    or self.measures.irSensor[left_id]   > 5.0\
        #    or self.measures.irSensor[right_id]  > 5.0\
        #    or self.measures.irSensor[back_id]   > 5.0:
        #     print('Rotate left')
        #     self.driveMotors(-0.1,+0.1)
        # elif self.measures.irSensor[left_id]> 2.7:
        #     print('Rotate slowly right')
        #     self.driveMotors(0.1,0.0)
        # elif self.measures.irSensor[right_id]> 2.7:
        #     print('Rotate slowly left')
        #     self.driveMotors(0.0,0.1)
        # else:
        #     print('Go')
        #     self.driveMotors(0.1,0.1)

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
    angles = [0.0, 90.0, -90.0, 180.0]
    base_speed = 0.1
    P = 0
    I = 0
    D = 0

    rob=MyRob(rob_name, pos, angles, host, 
        base_speed=base_speed,
        P=P,
        I=I,
        D=D)

    if mapc != None:
        rob.setMap(mapc.labMap)
        rob.printMap()
    
    rob.run()
