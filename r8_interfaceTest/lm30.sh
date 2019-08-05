#!/bin/bash
#SBATCH -p normal
#SBATCH -n 48
#SBATCH -t 96:00:00

# SBATCH -p short
# SBATCH --nodes=4
# SBATCH --constraint=infiniband,avx2
# SBATCH --constraint=infiniband
# SBATCH --tasks-per-node 12

echo "********** CPU-INFO **********"
lscpu
echo "********** XXXXXXXX **********"
# echo "********** Listing TMPDIR **********"

# ls -l
export IMAGE_STORE='/home/alaik/singularityImages'
export UW_ENABLE_TIMING='1'
#export IMAGE_VERSION='2.5.1b_magnus'
#export IMAGE_VERSION='2.7.0_prerelease'
#export IMAGE_VERSION='dev'
export IMAGE_VERSION='2.7.1b'
echo 'UW_VERSION: '$IMAGE_VERSION


echo "********** Run Started **********"

srun -n 48 singularity exec --pwd $PWD $IMAGE_STORE/underworld2-$IMAGE_VERSION.simg  python lm30.py

echo "********** XXXXXXXXXXX **********"

wait
