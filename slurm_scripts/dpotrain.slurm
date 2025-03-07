#!/bin/bash

#SBATCH --job-name=dpoTrainLlama8b
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/dpoTrainLlama8b.txt
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/dpoTrainLlama8b.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus


source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

srun python scripts/dpotrain.py \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --train_dataset "/sorgin1/users/jbarrutia006/viper/PrefDatasets/dpo_dataset_single_train.arrow" \
    --dev_dataset "/sorgin1/users/jbarrutia006/viper/PrefDatasets/dpo_dataset_single_train.arrow" \
    --output_dir "./dpo_llama3" \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --learning_rate 1e-5 \
    --epochs 3 \
    --max_steps 14000 \
    --logging_steps 10 \
    --eval_steps 200 \
    --save_steps 500 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --device "cuda"