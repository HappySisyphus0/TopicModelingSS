#!/bin/bash
#SBATCH -JClean_speech_1
#SBATCH -N1
#SBATCH --gres=gpu:V100:1
#SBATCH -CV100-32GB 
#SBATCH --mem-per-gpu=32G                           # Memory per gpu
#SBATCH -qembers
#SBATCH --time= 1:00:00
#SBATCH --output=Cleaning_dt.log
#SBATCH --error=Cleaning_dt.log
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=smudireddy3@gatech.edu            # e-mail address for notifications


source ~/myenv/bin/activate
cd ~/p-kgoyal-8-0/

srun python CleanChunks.py


