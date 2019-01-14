#!/bin/bash
#SBATCH --nodes=2
#SBATCH -p short
#SBATCH --tasks-per-node 12
#SBATCH --time=00:04:40

echo "********** Run Started **********"

# srun -n 24 singularity exec --pwd $PWD /home/alaik/uw/underworld2-dev.simg  python forTesting-v0.0.8.py
srun -n 24 singularity exec --pwd $PWD /home/alaik/uw/underworld2-2.5.1b_magnus.simg  python forTesting-v0.0.8.py
echo "********** XXXXXXXXXXX **********"


wait
