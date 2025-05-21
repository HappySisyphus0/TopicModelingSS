#!/bin/bash
#SBATCH -Likelihood_of_Prompted_B
#SBATCH -Agts-kgoyal8
#SBATCH -N1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem-per-gpu=40G                           # Memory per gpu
#SBATCH -qembers
#SBATCH --time= 10:00:00
#SBATCH --output=Bad_prompt_output.log
#SBATCH --error=Bad_prompt_error.log
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=smudireddy3@gatech.edu            # e-mail address for notifications


source ~/myenv/bin/activate
cd ~/p-kgoyal-8-0/

srun python Bad_Prompt_Likelihoods.py


