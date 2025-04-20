#!/bin/bash

#SBATCH --job-name=dpoTrainLlama
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/dpoTrainLlamaSp.txt
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/dpoTrainLlamaSp.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus


source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

python scripts/dpo/dpotrain.py \
    --project_name "viperDPO" \
    --run_name "New llama specific dataset 4,661 instances. Increase beta" \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --train_dataset "/sorgin1/users/jbarrutia006/viper/syntData/PrefDatasets/dpo_dataset_llama_train.arrow" \
    --dev_dataset "/sorgin1/users/jbarrutia006/viper/syntData/PrefDatasets/dpo_dataset_single_dev.arrow" \
    --output_dir "./dpo_trained_models" \
    --batch_size 32 \
    --gradient_accumulation 1 \
    --learning_rate 1e-6 \
    --beta 0.1 \
    --weight_decay 0.01 \
    --epochs 4 \
    --max_steps -1 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 100 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --device "cuda"
