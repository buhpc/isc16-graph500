#!/bin/bash

# enter number of processes as 1st CL arg
mpirun-openmpi-mp -np $1 main config.ini