#!/bin/bash

# === SLURM job parameters as variables ===
JOB_NAME="codigo_GQA_dpo"
CPUS_PER_TASK=1
NTASKS_PER_NODE=1
MEMORY=18GB
GRES="gpu:1"
OUTPUT="/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/codex_llama318b-dpo.txt"
ERROR="/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/codex_llama318b-dpo.err"
MAIL_TYPE="ALL"
MAIL_USER="jbarrutia006@ikasle.ehu.eus"

# === Export environment variables if needed ===
export OMP_NUM_THREADS=$CPUS_PER_TASK

# === Launch the interactive job ===
exec -a "$JOB_NAME" srun \
  --job-name="$JOB_NAME" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --ntasks-per-node="$NTASKS_PER_NODE" \
  --mem="$MEMORY" \
  --gres="$GRES" \
  --mail-type="$MAIL_TYPE" \
  --mail-user="$MAIL_USER" \
  --pty bash -i
# === Activate the virtual environment ===
source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate
# === Export environment variables for the script ===