#!/bin/bash
# SBATCH -p short
#SBATCH --nodes=2
# SBATCH --constraint=infiniband,avx2
#SBATCH --tasks-per-node 12
#SBATCH --time=72:00:00

echo "********** CPU-INFO **********"
lscpu
echo "********** XXXXXXXX **********"
# echo "********** Listing TMPDIR **********"

# ls -l
export IMAGE_STORE='/home/alaik/singularityImages'
export IMAGE_VERSION='2.5.1b_magnus'
#export IMAGE_VERSION='2.7.0_prerelease'
#export IMAGE_VERSION='dev'
echo "UW_VERSION: "$IMAGE_VERSION


echo "********** Run Started **********"

srun -n 24 singularity exec --pwd $PWD $IMAGE_STORE/underworld2-$IMAGE_VERSION.simg  python iea2D-v0.0.8_ul.py

echo "********** XXXXXXXXXXX **********"

wait
