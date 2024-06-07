#!/usr/bin/env python3
from expTools import *

easypapOptions = {
    "-k ": ["mandel"],
    "-i ": [10],
    "-v ": ["omp_tiled"],
    "-s ": [512],
    "-th ": [2 ** i for i in range(0, 10)],
    "-tw ": [2 ** i for i in range(0, 10)],
    "-of ": ["heat-mandel.csv"]
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE=": ["dynamic", "static,1"],
    "OMP_NUM_THREADS=": [24],
    "OMP_PLACES=": ["cores"]
}
nbrun = 1

execute('./run ', ompICV, easypapOptions,
        nbrun, verbose=True, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1

ompICV = {
    "OMP_NUM_THREADS=": [1]
}

del easypapOptions["-th "]
del easypapOptions["-tw "]
easypapOptions["-v "] = ["seq"]

execute('./run ', ompICV, easypapOptions,
        nbrun=1, verbose=False, easyPath=".")
