#!/bin/sh
# SBATCH -p short
#SBATCH --constraint=infiniband
#SBATCH --nodes=2
#SBATCH --tasks-per-node 12
#SBATCH --time=12:00:00

echo "********** Run Started **********"

#srun -n 24 singularity exec --pwd $PWD /home/alaik/uw/underworld2-dev.simg  python forTesting-v0.0.8.py
srun -n 24 singularity exec --pwd $PWD /home/alaik/singularityImages/ar_uw27.simg  python forTesting-v0.0.8.py
echo "********** XXXXXXXXXXX **********"
wait
