#!/usr/bin/env python3
from expTools import *

easypapOptions = {
    "-k ": ["mandel"],
    "-i ": [10],
    "-v ": ["omp_tiled"],
    "-s ": [512],
    "-ts ": [8, 16, 32],
    "-of ": ["mandel.csv"]
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE=": ["static", "static,1", "dynamic"],
    "OMP_NUM_THREADS=": [1] + list(range(2, 49, 4)) + [47, 48]
}

nbrun = 3
# Lancement des experiences
execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")

del easypapOptions["-ts "]
easypapOptions["-th "] = [1]
easypapOptions["-tw "] = [32, 64, 128, 256, 512]

#execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1

easypapOptions = {
    "-k ": ["mandel"],
    "-i ": [10],
    "-v ": ["seq"],
    "-s ": [512],
}
ompICV = {"OMP_NUM_THREADS=": [1]}
#execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")
