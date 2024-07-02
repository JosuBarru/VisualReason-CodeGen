#!/bin/bash
#SBATCH --job-name=proba_eneko
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_output.txt
#SBATCH --error=slurm_output.err
#SBATCH --mail-type=all
#SBATCH --mail-user=eamor002@ikasle.ehu.eus

source /gaueko0/users/eamor002/venvs/Tximista_GPT/bin/activate

export CODEX_QUANTIZED=1
export LOAD_MODELS=0
export DATASET='okvqa'
export EXEC_MODE='codex'

srun python src/main_project.py