#!/bin/bash

#SBATCH --job-name=gqa_test_inst_NF
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/%x-%A.log


source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

MODELS=("llama31Q" "codellama" "qwen25-inst")
CHECKPOINTS=("" "" "")

NUM_JOBS=${#MODELS[@]}

for (( i=0; i<${NUM_JOBS}; i++ )); do
  MODEL_NAME="${MODELS[i]}"
  CURRENT_CHECKPOINT="${CHECKPOINTS[i]}"

  echo "===================================================================="
  echo "Running evaluation for model: ${MODEL_NAME}"
  echo "Checkpoint: ${CURRENT_CHECKPOINT}"
  echo "Timestamp: $(date)"
  echo "===================================================================="

  export LOAD_MODELS=0
  export DATASET='gqa'
  export EXEC_MODE='codex'
  export CODEX_MODEL=${MODEL_NAME}       
  export CHECKPOINT="${CURRENT_CHECKPOINT}" 
  export PARTITION='testdev'
  export BATCHSIZE=8
  export TEMP=0
  export NUMINST=

  echo "Starting first run (codex mode) for ${MODEL_NAME} with checkpoint ${CURRENT_CHECKPOINT}"
  echo "Command: srun python src/main_project_slurm.py"
  echo "Environment:"
  echo "  LOAD_MODELS=${LOAD_MODELS}"
  echo "  DATASET=${DATASET}"
  echo "  EXEC_MODE=${EXEC_MODE}"
  echo "  CODEX_MODEL=${CODEX_MODEL}"
  echo "  CHECKPOINT=${CHECKPOINT}" 
  echo "  PARTITION=${PARTITION}"
  echo "  BATCHSIZE=${BATCHSIZE}"
  echo "  TEMP=${TEMP}"
  echo "  NUMINST=${NUMINST}"

  srun python src/main_project_slurm.py
  echo "First run for ${MODEL_NAME} completed."

  shopt -s nocaseglob

  CSV_FILE_PATH_PATTERN="results/gqa/codex_results/${PARTITION}/${CODEX_MODEL}*.csv"
  echo "Looking for CSV file with pattern: ${CSV_FILE_PATH_PATTERN}"
  
  LATEST_CSV=$(ls -t ${CSV_FILE_PATH_PATTERN} 2>/dev/null | head -n 1)
  shopt -u nocaseglob

  if [ -z "$LATEST_CSV" ]; then
      echo "Error: No file matching pattern '${CSV_FILE_PATH_PATTERN}' was found for model ${MODEL_NAME} and checkpoint ${CURRENT_CHECKPOINT}." >&2
      echo "Skipping second run for ${MODEL_NAME} due to missing CSV." >&2
      echo "--------------------------------------------------------------------"
      continue 
  fi
  
  export CODE=$(basename "${LATEST_CSV}")
  echo "Found CSV file: ${CODE} for model ${MODEL_NAME}"

  export LOAD_MODELS=1
  export EXEC_MODE='cache'
  export BATCHSIZE=32

  echo "Starting second run (cache mode) for ${MODEL_NAME}"
  echo "Command: srun python src/main_project_slurm.py"
  echo "Environment:"
  echo "  LOAD_MODELS=${LOAD_MODELS}"
  echo "  DATASET=${DATASET}" 
  echo "  EXEC_MODE=${EXEC_MODE}"
  echo "  CODEX_MODEL=${CODEX_MODEL}" 
  echo "  CHECKPOINT=${CHECKPOINT}" 
  echo "  CODE_FILE_TO_EXECUTE=${CODE}" 
  echo "  PARTITION=${PARTITION}"
  echo "  BATCHSIZE=${BATCHSIZE}"
  
  srun python src/main_project_slurm.py
  echo "Second run for ${MODEL_NAME} completed."
  echo "--------------------------------------------------------------------"

  echo "Finished evaluation for model: ${MODEL_NAME} with checkpoint ${CURRENT_CHECKPOINT}"
  echo "===================================================================="
done

echo "All model and checkpoint combinations processed."
echo "Job finished at: $(date)"