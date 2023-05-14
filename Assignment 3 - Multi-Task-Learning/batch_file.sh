#!/bin/bash
#SBATCH --job-name=DL_Assignment_3 # Job name
#SBATCH --partition=gpu2 #Partition Name
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run a single task
#SBATCH --cpus-per-task=1# Number of CPU cores per task
#SBATCH --gres=gpu
#

sleep 3
echo "Executing the job by B20AI052"
module load python/3.8

sleep 3
nvidia-smi

sleep 3
python3 Assignment3.py

