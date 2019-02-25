#!/bin/bash
#SBATCH --nodes=2
#SBATCH -p short
# SBATCH --constraint=infiniband,avx2
#SBATCH --tasks-per-node 16
#SBATCH --time=5:00

# glxinfo
# echo "********** Copying Scripts **********"
# # UWScDir=`mktemp -d /scratch/uwalaik.XXXX`
# UWScDir="$TMPDIR"/alaiktest2
# mkdir "$UWScDir"
# echo "$UWScDir"
# cd "$UWScDir"
# pwd
#
# cp  $HOME/uw/*.py ./
# cp -r $HOME/uw/scaling ./
# cp  $HOME/uw/*.simg ./
echo "********** CPU-INFO **********"
lscpu
echo "********** XXXXXXXX **********"
# echo "********** Listing TMPDIR **********"

# ls -l
echo "********** Run Started **********"
# srun -n 12 singularity exec --pwd $UWScDir  underworld2-dev.simg  python iea2D-v0.0.8LR.py
srun -n 32 singularity exec --pwd $PWD underworld2-2.7.0_prerelease.simg  python Subduction-SouthAmerica-20181129-isoThermalMantle.py
# srun -n 24 singularity exec --pwd $PWD underworld2-dev.simg  python iea2D-v0.0.8_hr.py

# srun -n 64 singularity exec --pwd $PWD underworld2-dev.simg  python iea2D-v0.0.8_hr.py
echo "********** XXXXXXXXXXX **********"

# cat opTeLR_gpu/*.log
# cp -r opTeLR_gpuMR/  $HOME/uw/opD

wait
