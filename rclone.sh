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
rclone -v sync /home/alaik/uw/ gdalk:/LisaBackup

wait
