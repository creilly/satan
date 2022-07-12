#!/bin/bash

module purge
module load intel intel-mkl intel-mpi python
srun -n 28 python optimize.py