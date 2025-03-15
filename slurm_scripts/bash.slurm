#!/bin/bash

#SBATCH --job-name=bash
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/bash.txt
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/bash.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus

source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate
export CUDA_VISIBLE_DEVICES=0
srun python -c 'import torch; print("Visible GPUs:", torch.cuda.device_count())'