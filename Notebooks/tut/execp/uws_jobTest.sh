#!/bin/bash
#SBATCH --nodes=2
# SBATCH -p short
# SBATCH --constraint=infiniband,avx2
#SBATCH --tasks-per-node 12
#SBATCH --time=72:00:40

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
# srun -n 48 singularity exec --pwd $PWD underworld2-2.7.0_prerelease.simg  python iea2D-v0.0.8_hr.py
# srun -n 24 singularity exec --pwd $PWD /home/alaik/uw/underworld2-2.5.1b_magnus.simg  python iea2D-v0.0.8_hr.py
srun -n 24 singularity exec --pwd $PWD /home/alaik/uw/underworld2-dev.simg  python iea2D-v0.0.8_hr.py
# srun -n 64 singularity exec --pwd $PWD underworld2-dev.simg  python iea2D-v0.0.8_hr.py
echo "********** XXXXXXXXXXX **********"

# cat opTeLR_gpu/*.log
# cp -r opTeLR_gpuMR/  $HOME/uw/opD

wait
