#!/bin/bash

# ARGS="--param ../Labs/rmi-2122/C1-config.xml"
# ARGS+=" --lab ../Labs/rmi-2122/C1-lab.xml"
# ARGS+=" --grid ../Labs/rmi-2122/C1-grid.xml"
# ARGS+=" --scoring 1"

SIM_PATH="./ciberRatoTools"

$SIM_PATH/simulator/simulator --param $SIM_PATH/Labs/rmi-2122/C1-config.xml --lab $SIM_PATH/Labs/rmi-2122/C1-lab.xml --grid $SIM_PATH/Labs/rmi-2122/C1-grid.xml --scoring 1