#!/bin/bash
#SBATCH -Likelihood_of_Prompted_G
#SBATCH -Agts-kgoyal8
#SBATCH -N1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem-per-gpu=40G                           # Memory per gpu
#SBATCH -qembers
#SBATCH --time= 10:00:00
#SBATCH --output=Good_prompt_output.log
#SBATCH --error=Good_prompt_error.log
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=smudireddy3@gatech.edu            # e-mail address for notifications


source ~/myenv/bin/activate
cd ~/p-kgoyal-8-0/

srun python Good_Prompt_Likelihoods.py


