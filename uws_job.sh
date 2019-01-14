#!/bin/bash
#SBATCH --nodes=4
#SBATCH --constraint=infiniband
#SBATCH --tasks-per-node 12

# SBATCH --constraint=avx2
# SBATCH --nodes=3
# SBATCH --tasks-per-node 16
#SBATCH --time=48:00:00

echo "********** Copying Scripts **********"
# UWScDir=`mktemp -d /scratch/uwalaik.XXXX`
# UWScDir=$TMPDIR/alaik_test
# mkdir UWScDir
# echo $UWScDir
# cd $UWScDir
#
# cp  $HOME/uw/*.py ./
# cp  $HOME/uw/*.simg ./
ls
echo "********** Run Started **********"
srun -n 48 singularity exec  --pwd $PWD underworld2-dev.simg  python iea2D-v0.0.8.py
# cp -r opTe  $HOME/uw/opTe

wait
