#!/bin/bash

#SBATCH --job-name=codigo_refcoco
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/refcoco_slurm.txt
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/refcoco_slurm.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus

source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

export CODEX_QUANTIZED=1
export LOAD_MODELS=0
export DATASET='refcoco'
export EXEC_MODE='codex'

srun python main_project.py