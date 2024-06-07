#!/usr/bin/env python3
from expTools import *

options = {
    "--kernel ": ["spin"],
    "--variant ": ["omp"],
    "--iterations ": [20],
    "--size ": [1024],
    "--tile-size ": [8,32,128]
}
# Pour renseigner l'option '-of' il faut donner le chemin depuis le répertoire easypap
options["-of "] = ["spin.csv"]

ompenv = {
    "OMP_NUM_THREADS=": list(range(4, 49, 4)),
    "OMP_SCHEDULE=": ["static", "dynamic"]
}
# Lancement des experiences
execute('./run ', ompenv, options, nbrun=3, verbose=False, easyPath=".")

# Version séquentielle
options["--variant "] = ["seq"]
del options["--tile-size "]
ompenv = {"OMP_NUM_THREADS=": [1]}
execute('./run', ompenv, options, nbrun=1, verbose=False, easyPath=".")
