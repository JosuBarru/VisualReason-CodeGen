#!/bin/bash

#SBATCH --job-name=sftTrain-base
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/%x.txt
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/%x.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus

# Activate your virtual environment
source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

# Run your SFT script
srun python scripts/sft/sftrain-base.py \
    --project_name "viperSFT" \
    --run_name "Llama base with SFT dataset. 4 epochs. Lr 1e-4" \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --train_dataset_sft "/sorgin1/users/jbarrutia006/viper/syntData/SFTDatasets/sft_dataset_train.arrow" \
    --dev_dataset_sft "/sorgin1/users/jbarrutia006/viper/syntData/SFTDatasets/sft_dataset_dev.arrow" \
    --output_dir "./sft_trained_models" \
    --batch_size 32 \
    --gradient_accumulation 1 \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --epochs 4 \
    --max_steps -1 \
    --logging_steps 20 \
    --eval_steps 50 \
    --save_steps 50 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --device "cuda" \
