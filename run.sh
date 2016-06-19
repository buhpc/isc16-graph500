#!/bin/bash

# enter number of processes as 1st CL arg, scale as 2nd and edge factor as 3rd
mpirun -np $1 main config.ini $2 $3
