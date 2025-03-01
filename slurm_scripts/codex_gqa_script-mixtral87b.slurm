#!/bin/bash

#SBATCH --job-name=codigo_GQA_mixtral
#SBATCH -D .
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=36GB
#SBATCH --gres=gpu:2
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/codex_mixtral.txt
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/codex_mixtral.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus


source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

export LOAD_MODELS=0
export DATASET='gqa'
export EXEC_MODE='codex'
export CODEX_MODEL='mixtral87b'
export TRAIN=True
export BATCHSIZE=8
export NUMINST=8

srun python src/main_project_slurm.py