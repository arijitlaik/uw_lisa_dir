#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00


ip addr
echo "********** Run Started **********"
rclone copy /home/alaik/uw/r8_interfaceTest/ gdalk:/EXPSET-n/Lisa/r8_interfaceTest/

wait
