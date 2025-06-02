#!/bin/bash

#SBATCH --job-name=test_GQA
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/%x.txt
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/%x.err

#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus

source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

export LOAD_MODELS=0
export DATASET='gqa'
export EXEC_MODE='codex'
export CODEX_MODEL='qwen25'
export PARTITION='testdev'
export BATCHSIZE=8
export TEMP=0
export NUMINST=
export CHECKPOINT='dpo_trained_models/qwen-05-28_22-37-13/checkpoint-2100'

srun python src/main_project_slurm.py

shopt -s nocaseglob

export CODE=$(basename "$(ls -t results/gqa/codex_results/${PARTITION}/${CODEX_MODEL}*.csv | head -n 1)")

shopt -u nocaseglob

if [ -z "$CODE" ]; then
    echo "Error: No file matching pattern '${CODEX_MODEL}*.csv' was found."
    exit 1
fi

export LOAD_MODELS=1
export EXEC_MODE='cache'
export BATCHSIZE=32

srun python src/main_project_slurm.py
