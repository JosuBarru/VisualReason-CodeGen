#!/bin/bash

#SBATCH --job-name=cod_eval_gqa3
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00
# Define a single output file for both stdout and stderr
# %x will be replaced by the job name (cod_eval_gqa3)
# %j will be replaced by the job ID (if you prefer that for uniqueness)
# Using %x as per your original script for consistency in naming.
# All output and errors from this script will go here.
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/general_eval_all_models_%x.log
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/general_eval_all_models_%x.log


source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

# Define the array of models
MODELS=("llama31Q_Base" "codellama_base" "qwen25")

# Loop through each model
for MODEL_NAME in "${MODELS[@]}"
do
  echo "===================================================================="
  echo "Running evaluation for model: ${MODEL_NAME}"
  echo "Timestamp: $(date)"
  echo "===================================================================="

  # --- Environment variables for the current model ---
  export LOAD_MODELS=0
  export DATASET='gqa'
  export EXEC_MODE='codex'
  export CODEX_MODEL=${MODEL_NAME} # Set the current model
  export PARTITION='val'
  export BATCHSIZE=8
  export TEMP=0
  export NUMINST=
  export CHECKPOINT=''

  echo "Starting first run (codex mode) for ${MODEL_NAME}"
  echo "Command: srun python src/main_project_slurm.py"
  echo "Environment:"
  echo "  LOAD_MODELS=${LOAD_MODELS}"
  echo "  DATASET=${DATASET}"
  echo "  EXEC_MODE=${EXEC_MODE}"
  echo "  CODEX_MODEL=${CODEX_MODEL}"
  echo "  PARTITION=${PARTITION}"
  echo "  BATCHSIZE=${BATCHSIZE}"
  echo "  TEMP=${TEMP}"
  echo "  NUMINST=${NUMINST}"
  echo "  CHECKPOINT=${CHECKPOINT}"
  # srun will inherit the job's stdout and stderr, which are set by #SBATCH directives
  srun python src/main_project_slurm.py
  echo "First run for ${MODEL_NAME} completed."

  shopt -s nocaseglob
  # --- Construct the path for the CSV file based on the current model ---
  CSV_FILE_PATH_PATTERN="results/gqa/codex_results/${PARTITION}/${CODEX_MODEL}*.csv"
  echo "Looking for CSV file with pattern: ${CSV_FILE_PATH_PATTERN}"
  
  # Use a subshell and error checking for ls to avoid script exit on no match with set -e
  LATEST_CSV=$(ls -t ${CSV_FILE_PATH_PATTERN} 2>/dev/null | head -n 1)
  shopt -u nocaseglob

  if [ -z "$LATEST_CSV" ]; then
      # Send error messages to stderr, which will be captured in the main log file
      echo "Error: No file matching pattern '${CSV_FILE_PATH_PATTERN}' was found for model ${MODEL_NAME}." >&2
      echo "Skipping second run for ${MODEL_NAME} due to missing CSV." >&2
      echo "--------------------------------------------------------------------"
      continue # Skip to the next model
  fi
  
  export CODE=$(basename "${LATEST_CSV}")
  echo "Found CSV file: ${CODE} for model ${MODEL_NAME}"

  export LOAD_MODELS=1
  export EXEC_MODE='cache'
  export BATCHSIZE=32 # This BATCHSIZE is for the second run

  echo "Starting second run (cache mode) for ${MODEL_NAME}"
  echo "Command: srun python src/main_project_slurm.py"
  echo "Environment:"
  echo "  LOAD_MODELS=${LOAD_MODELS}"
  echo "  DATASET=${DATASET}" # Assuming dataset is the same
  echo "  EXEC_MODE=${EXEC_MODE}"
  echo "  CODEX_MODEL=${CODEX_MODEL}" # Assuming CODEX_MODEL is still relevant or used to find CODE
  echo "  CODE_FILE_TO_EXECUTE=${CODE}" # The actual file to execute is now in CODE
  echo "  PARTITION=${PARTITION}"
  echo "  BATCHSIZE=${BATCHSIZE}"
  # srun will inherit the job's stdout and stderr
  srun python src/main_project_slurm.py
  echo "Second run for ${MODEL_NAME} completed."
  echo "--------------------------------------------------------------------"

  echo "Finished evaluation for model: ${MODEL_NAME}"
  echo "===================================================================="
done

echo "All models processed."
echo "Job finished at: $(date)"