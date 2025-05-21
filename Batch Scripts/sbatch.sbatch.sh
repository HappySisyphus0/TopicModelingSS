#!/bin/bash
#SBATCH -JPGUExample
#SBATCH -Agts-kgoyal8
#SBATCH -N1--gres=gpu:V100:1
#SBATCH --mem-per-gpu=12G                           # Memory per gpu
#SBATCH -qembers
#SBATCH --time=6:00:00
#SBATCH --output=job_output.log
#SBATCH --error=job_error.log
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=smudireddy3@gatech.edu            # e-mail address for notifications


source ~/myenv/bin/activate
cd ~/p-kgoyal-8-0/

srun python Sample_Likelihoods.py


