#!/bin/bash

#SBATCH --job-name=test_GQA
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/%x-%A.txt

#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus

source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

export LOAD_MODELS=0
export DATASET='gqa'
export EXEC_MODE='codex'
export CODEX_MODEL='llama31Q'
export PARTITION='testdev'
export BATCHSIZE=8
export TEMP=0
export NUMINST=
export CHECKPOINT='sft_trained_models/llama3.1-lr-1e-4__05-25_12-44-42/checkpoint-650'

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
