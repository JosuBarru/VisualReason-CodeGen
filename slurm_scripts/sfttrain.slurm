#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/sft
#SBATCH --error=.slurm/alpaca-lora-7b.err

