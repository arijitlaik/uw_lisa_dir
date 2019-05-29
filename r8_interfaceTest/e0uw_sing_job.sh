#!/bin/bash
#SBATCH -p normal
#SBATCH -n 48
#SBATCH -t 96:00:00

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

srun -n 48 singularity exec --pwd $PWD $IMAGE_STORE/underworld2-$IMAGE_VERSION.simg  python R8iea2D_a_FaBa_e0_nlLM_thickUC_ClikeOp.py

echo "********** XXXXXXXXXXX **********"

wait
