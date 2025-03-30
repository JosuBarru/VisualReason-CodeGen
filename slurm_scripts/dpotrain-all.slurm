#!/bin/bash

#SBATCH --job-name=dpoTrainQwen
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/dpoTrainQwen.txt
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/dpoTrainQwen.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus


source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

srun python scripts/dpo/dpotrain.py \
    --project_name "viperDPO" \
    --run_name "Qwen All" \
    --model_name "Qwen/Qwen2.5-Math-7B" \
    --train_dataset "/sorgin1/users/jbarrutia006/viper/syntData/PrefDatasets/dpo_dataset_all_train.arrow" \
    --dev_dataset "/sorgin1/users/jbarrutia006/viper/syntData/PrefDatasets/dpo_dataset_single_dev.arrow" \
    --output_dir "./dpo_trained_models" \
    --batch_size 16 \
    --gradient_accumulation 1 \
    --learning_rate 5e-5 \
    --beta 0.1 \
    --weight_decay 0.01 \
    --epochs 2 \
    --max_steps -1 \
    --logging_steps 20 \
    --eval_steps 300 \
    --save_steps 300 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --device "cuda"
