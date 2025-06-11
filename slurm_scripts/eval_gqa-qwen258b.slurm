#!/bin/bash

#SBATCH --job-name=eval_GQA_qwen
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/eval_output_qwen.txt
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/eval_output_qwen.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus


source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate


export LOAD_MODELS=0
export DATASET='gqa'
export EXEC_MODE='codex'
export CODEX_MODEL='qwen25_inst'
export PARTITION='val'
export BATCHSIZE=16
export TEMP=0
export NUMINST=
export CHECKPOINT=''

export CODE=$(basename "$(ls -t results/gqa/codex_results/${PARTITION}/qwen25_inst.csv | head -n 1)")
export LOAD_MODELS=1
export EXEC_MODE='cache'
export BATCHSIZE=32


srun python src/main_project_slurm.py
