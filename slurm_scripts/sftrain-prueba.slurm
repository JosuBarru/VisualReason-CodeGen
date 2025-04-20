#!/bin/bash

#SBATCH --job-name=sftTrain
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/sftTrain-pruebas.txt
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/sftTrain-pruebas.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus

# Activate your virtual environment
source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

# Run your SFT script
srun python scripts/sft/sftrain.py \
    --project_name "viperSFT" \
    --run_name "Llama with SFT dataset. 1 epoch" \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --train_dataset_sft "/sorgin1/users/jbarrutia006/viper/syntData/SFTDatasets/sft_dataset_train.arrow" \
    --dev_dataset_sft "/sorgin1/users/jbarrutia006/viper/syntData/SFTDatasets/sft_dataset_dev.arrow" \
    --dev_dataset_dpo "/sorgin1/users/jbarrutia006/viper/syntData/PrefDatasets/dpo_dataset_single_dev.arrow" \
    --output_dir "./sft_trained_models" \
    --batch_size 32 \
    --gradient_accumulation 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --epochs 1 \
    --max_steps -1 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 100 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --device "cuda" \
