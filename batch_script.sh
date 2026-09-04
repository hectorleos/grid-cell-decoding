#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --output=logs/simulation_%j.out
#SBATCH --error=logs/simulation_%j.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
  
srun python simulation_script.py
 