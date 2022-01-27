#!/usr/bin/env python3
import sys
from croblink import *
from math import *
import xml.etree.ElementTree as ET

"""
JS Imports
"""
import time
import numpy as np

CELLROWS=7
CELLCOLS=14

class PID:
    """Ivmech PID Controller is simple implementation of a Proportional-Integral-Derivative (PID) Controller in the Python Programming Language.
    More information about PID Controller: http://en.wikipedia.org/wiki/PID_controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0, current_time=None):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
           :align:   center
           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
        """
        error = self.SetPoint - feedback_value

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time

class MyRob(CRobLinkAngs):
    def __init__(self, 
                rob_name, 
                rob_id, 
                angles, 
                host, 
                base_speed=0.1, 
                P=0, 
                I=0, 
                D=0, 
                set_point=0.0,
                windup=20.0,
                weights=[1.0]*NUM_IR_SENSORS+[0],
                mem_size = 1,
                in_eval= False):

        CRobLinkAngs.__init__(self, rob_name, rob_id, angles, host)

        self.sample_time = 1e-3 # seconds
        self.base_speed = base_speed
        self.P = P
        self.I = I
        self.D = D
        self.set_point = set_point
        self.windup = windup
        self.weights = weights #w0, w1, w2, w3 and Ksr
        self.in_eval = in_eval

        self.memory = np.zeros(mem_size)

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

        # initialize PID
        self.pid = PID(self.P, self.I, self.D)
        self.pid.SetPoint = self.set_point
        self.pid.setSampleTime(self.sample_time)
        self.pid.setWindup(self.windup)

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
            mv = np.sum(np.multiply(self.weights[:4], self.measures.irSensor))
            self.pid.update(mv)
            
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
        
        self.memory = np.roll(self.memory, -1)
        self.memory[-1] = self.pid.output
        deduction = np.mean(np.abs(self.memory)) * self.weights[-1]

        coefficients = np.array([[1, 1], [1, -1]])
        results = np.array([2 * self.base_speed, self.pid.output])
        v_right, v_left = np.linalg.solve(coefficients, results) + deduction
        self.driveMotors(v_left, v_right)

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

    solution = [.880048, -1.78629, 0.001976, -0.375373, 22.163, 52, 90, -0.435247, -0.296408, 0,  0]
    
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
            weights=weights)

    if mapc != None:
        rob.setMap(mapc.labMap)
        rob.printMap()

    rob.run()


