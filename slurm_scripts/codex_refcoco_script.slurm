#!/bin/bash
#SBATCH --job-name=GQA_CODEX
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_output_gqa.txt
#SBATCH --error=slurm_output_gqa.err
#SBATCH --mail-type=all
#SBATCH --mail-user=eamor002@ikasle.ehu.eus

source /gaueko0/users/eamor002/venvs/Tximista_GPT/bin/activate

export CODEX_QUANTIZED=1
export LOAD_MODELS=0
export DATASET='refcoco'
export EXEC_MODE='codex'

srun python src/main_project.py