#!/bin/bash
#SBATCH --nodes=3
#SBATCH --constraint=avx2
#SBATCH --tasks-per-node 16
#SBATCH --time=24:00:00
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
echo "********** CPU-INFO**********"
lscpu

# echo "********** Listing TMPDIR **********"

# ls -l
echo "********** Run Started **********"
# srun -n 12 singularity exec --pwd $UWScDir  underworld2-dev.simg  python iea2D-v0.0.8LR.py
# srun -n 48 singularity exec --pwd $PWD underworld2-dev.simg  python iea2D-v0.0.8.py
srun -n 48 singularity exec --pwd $PWD uwgeodynamics-dev.simg  python3 Tutorial_10_Thrust_Wedges.py

# cat opTeLR_gpu/*.log
# cp -r opTeLR_gpuMR/  $HOME/uw/opD

wait
