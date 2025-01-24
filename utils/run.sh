#!/bin/bash

#SBATCH --array               0-9
#SBATCH --cpus-per-task       12
#SBATCH --time                1-14:22:46

module load anaconda
source activate torch

python calc_conductance.py
