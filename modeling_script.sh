#!/bin/bash

#SBATCH --gres=gpu:1
##SBATCH --partition=v100_sxm2_4
#SBATCH --nodes=1
#SBATCH --job-name=modeling
#SBATCH --mem=100GB
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out
#SBATCH --time=7-00:00:00

#command line argument
cd /scratch/fg746/capstone/Capstone
source setup.sh
export MPLBACKEND="pdf"
python -u modeling.py 5000 64 25 0.001 'full'
