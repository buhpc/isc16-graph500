#!/bin/bash

# enter number of processes as 1st CL arg, scale as 2nd and edge factor as 3rd
time mpirun -x LD_LIBRARY_PATH --hostfile hostfile --rankfile rankfile -np $1 main config.ini $2 $3

# get results
#for node in `cat hostfile`; do
#  scp ubuntu@$node:isc16-graph500/log/* ./log/
#done

#for file in `ls log`; do
#  outputs=$outputs" log/"$file
#done
#echo ""

#python scripts/parseResults.py $outputs
