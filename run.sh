#!/bin/bash

# enter number of processes as 1st CL arg, scale as 2nd and edge factor as 3rd
/opt/arm/openmpi-1.10.2_Cortex-A57_Ubuntu-14.04_aarch64-linux/bin/mpirun -x LD_LIBRARY_PATH --hostfile /home/ubuntu/isc16-graph500/hostfile --rankfile /home/ubuntu/isc16-graph500/rankfile -np $1 /home/ubuntu/isc16-graph500/main /home/ubuntu/isc16-graph500/config.ini $2 $3

# get results
for node in `cat hostfile`; do
  scp ubuntu@$node:isc16-graph500/log/* ./log/
done

for file in `ls log`; do
  outputs=$outputs" log/"$file
done
echo ""

python scripts/parseResults.py $outputs
