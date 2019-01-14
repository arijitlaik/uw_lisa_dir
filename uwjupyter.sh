#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=infiniband
#SBATCH --tasks-per-node 16

# SBATCH --constraint=avx2
# SBATCH --nodes=3
# SBATCH --tasks-per-node 16
#SBATCH --time=48:00:00


ip addr
echo "********** Run Started **********"
singularity exec  --pwd $PWD underworld2-dev.simg  jupyter notebook --no-browser

# cp -r opTe  $HOME/uw/opTe

wait
