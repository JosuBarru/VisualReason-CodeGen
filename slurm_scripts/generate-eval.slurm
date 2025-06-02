#!/bin/bash

#SBATCH --job-name=generate_eval
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/%x-%A.log


#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus

source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

export LOAD_MODELS=0
export DATASET='gqa'
export EXEC_MODE='codex'
export CODEX_MODEL='codellama'
export PARTITION='val'
export BATCHSIZE=8
export TEMP=0
export NUMINST=
export CHECKPOINT=''

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
