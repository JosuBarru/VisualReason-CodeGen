#!/bin/bash

#SBATCH --job-name=verify_length
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/verify_sft_length.txt
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/verify_sft_length.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus


source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

srun python scripts/sft/verify_sft_length.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --train_dataset "/sorgin1/users/jbarrutia006/viper/syntData/SFTDatasets/sft_dataset_train.arrow" \
    --dev_dataset "/sorgin1/users/jbarrutia006/viper/syntData/SFTDatasets/sft_dataset_dev.arrow" \
    --dir_plot "/sorgin1/users/jbarrutia006/viper/syntData/SFTDatasets"
