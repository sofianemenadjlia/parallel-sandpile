#!/usr/bin/env python3
from expTools import *

exec_list = [2**i for i in range (2, 6)]

easypapOptions = {
    "-k ": ["ssandPile"],
    # "-i ": [10],
    "-v ": ["ocl_hybrid"],
    "-o ": [""],
    "-s ": [2048],
    "-wt ": ["opt"],
    "-a ": ["alea"],
    "-th ": [32],
    "-tw ": [32],
    "-of ": ["hybrid.csv"]
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE=": ["static", "dynamic"],
    "OMP_NUM_THREADS=": [44, 46, 48], 
    "OMP_PLACES=": ["cores"],
    "TILEX=": [32],
    "TILEY=": [32] 

}

nbrun = 2
for th in exec_list:
    for tw in exec_list:
        easypapOptions["-th "] = [th]
        easypapOptions["-tw "] = [tw]
        ompICV["TILEX="] = [tw]
        ompICV["TILEY="] = [th]
        execute('./run ', ompICV, easypapOptions, nbrun, verbose=True, easyPath=".")

ompICV = {
    "OMP_NUM_THREADS=": [1]
}

del easypapOptions["-th "]
del easypapOptions["-tw "]
del easypapOptions["-o "]
easypapOptions["-v "] = ["seq"]


execute('./run ', ompICV, easypapOptions,
        nbrun=1, verbose=False, easyPath=".")