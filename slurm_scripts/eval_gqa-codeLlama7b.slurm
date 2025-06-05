#!/bin/bash

#SBATCH --job-name=eval_GQA_lcodellama
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/%x-%A.log


source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

export LOAD_MODELS=1
export DATASET='gqa'
export EXEC_MODE='cache'
export PARTITION='testdev'
export CODE='codellama___06-05_18-11.csv'
export BATCHSIZE=32
export NUMINST=
export TEMP=0


srun python src/main_project_slurm.py
