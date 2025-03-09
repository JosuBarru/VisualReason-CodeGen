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
    --project_name "viperDPO" \
    --run_name "Primera prueba A100" \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --train_dataset "/sorgin1/users/jbarrutia006/viper/PrefDatasets/dpo_dataset_single_train.arrow" \
    --dev_dataset "/sorgin1/users/jbarrutia006/viper/PrefDatasets/dpo_dataset_single_train.arrow" \
    --output_dir "./dpo_trained_models" \
    --batch_size 32 \
    --gradient_accumulation 1 \
    --learning_rate 1e-5 \
    --epochs 4 \
    --max_steps -1 \
    --logging_steps 40 \
    --eval_steps 100 \
    --save_steps 400 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --device "cuda"
