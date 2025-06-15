import os
import re
import pandas as pd
from datasets import Dataset, load_from_disk
from typing import List, Dict
from datasets import concatenate_datasets

# CHANGED: Updated paths to point to the new experimental DPO datasets
DPO_TRAIN_PATH = "/sorgin1/users/jbarrutia006/viper/syntData/PrefDatasets/dpo_dataset_single_train_experimental.arrow"
DPO_EVAL_PATH  = "/sorgin1/users/jbarrutia006/viper/syntData/PrefDatasets/dpo_dataset_single_dev_experimental.arrow"

SFT_OUTPUT_DIR = "/sorgin1/users/jbarrutia006/viper/syntData/SFTDatasets/"
os.makedirs(SFT_OUTPUT_DIR, exist_ok=True)

INPUT_FOLDER = "/sorgin1/users/jbarrutia006/viper/results/gqa/all/train"

def get_csv_files(directory: str) -> List[str]:
    """Return a list of CSV filenames in the given directory."""
    return [f for f in os.listdir(directory) if f.endswith('.csv')]

def select_multiple_csv_files(csv_files: List[str]) -> List[str]:
    """Interactively select multiple CSV files from the list."""
    selected_files = []
    print("Lista de archivos disponibles:")
    for idx, file in enumerate(csv_files):
        print(f"{idx}: {file}")
    while True:
        user_input = input("Selecciona el índice del archivo CSV (o escribe 'fin' para terminar): ").strip()
        if user_input.lower() == 'fin':
            break
        try:
            idx = int(user_input)
            if 0 <= idx < len(csv_files):
                selected_files.append(csv_files[idx])
            else:
                print("Índice inválido, intenta de nuevo.")
        except ValueError:
            print("Entrada no válida. Por favor, introduce un número o 'fin'.")
    return selected_files

def parse_model_name(file_name: str) -> str:
    """Extract model name from a filename like 'eval_modelName.csv'."""
    match = re.match(r'^eval_(.*)\.csv$', file_name)
    return match.group(1) if match else file_name

def load_csvs(file_paths: List[str]) -> pd.DataFrame:
    """Load multiple CSV files into a single pandas DataFrame."""
    df_list = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, engine='python')
        if len(df) > 0:
            df = df.iloc[:-1]
        base_name = os.path.basename(file_path)
        df["model_name"] = parse_model_name(base_name)
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df[combined_df['sample_id'].apply(lambda x: str(x).isnumeric())]
    return combined_df

def remove_function_header(code_str: str) -> str:
    """Remove function header lines from the given code string."""
    pattern = r"^def\s+execute_command_[^(]+\([^)]*\):\n"
    return re.sub(pattern, "", code_str, flags=re.MULTILINE)

def create_sft_instances(df: pd.DataFrame, sample_ids: List[str]) -> List[Dict[str, str]]:
    """Create SFT instances, now including sample_id."""
    instances = []
    grouped = df.groupby('sample_id')
    for sample_id in sample_ids:
        group = grouped.get_group(sample_id)
        correct_rows = group[group['accuracy'] == 1]
        if correct_rows.empty:
            continue
        chosen_row = correct_rows.sample(n=1).iloc[0]
        instances.append({
            'prompt': chosen_row['query'],
            'output': remove_function_header(chosen_row['code']),
            'model': chosen_row['model_name'],
            'sample_id': sample_id # CHANGED: Added sample_id to the instance dictionary
        })
    return instances

def main():
    dpo_train = load_from_disk(DPO_TRAIN_PATH)
    dpo_eval  = load_from_disk(DPO_EVAL_PATH)

    # The 'sample_id' column is now automatically kept
    sft_train = dpo_train.remove_columns(["rejected", "rejected_model"])
    sft_eval  = dpo_eval.remove_columns(["rejected", "rejected_model"])

    sft_train = sft_train.rename_column("chosen", "output")
    sft_eval  = sft_eval.rename_column("chosen", "output")

    csv_files = get_csv_files(INPUT_FOLDER)
    if not csv_files:
        print("No se encontraron archivos CSV en el directorio.")
        return

    selected_files = select_multiple_csv_files(csv_files)
    if not selected_files:
        print("No se seleccionó ningún archivo. Saliendo.")
        return

    file_paths = [os.path.join(INPUT_FOLDER, f) for f in selected_files]
    df = load_csvs(file_paths)

    valid_sample_ids = []
    grouped = df.groupby('sample_id')
    for sample_id in df['sample_id'].unique():
        group = grouped.get_group(sample_id)
        if len(group[group['accuracy'] == 1]) == len(group):
            valid_sample_ids.append(sample_id)

    print(f"Found {len(valid_sample_ids)} instances where all the models did well.")
    new_instances = create_sft_instances(df, valid_sample_ids)

    # CHANGED: Added 'sample_id' to the new Dataset
    new_dataset = Dataset.from_dict({
        "prompt": [p["prompt"] for p in new_instances],
        "output": [p["output"] for p in new_instances],
        "model": [p["model"] for p in new_instances],
        "sample_id": [p["sample_id"] for p in new_instances],
    })

    # Concatenate will now work correctly as both datasets have the sample_id column
    all_train = concatenate_datasets([sft_train, new_dataset])

    # CHANGED: Updated output filenames
    sft_train_output_path = os.path.join(SFT_OUTPUT_DIR, "sft_dataset_train_experimental.arrow")
    sft_eval_output_path = os.path.join(SFT_OUTPUT_DIR, "sft_dataset_eval_experimental.arrow")
    
    all_train.save_to_disk(sft_train_output_path)
    sft_eval.save_to_disk(sft_eval_output_path)

    print("\nNew SFT datasets written to", SFT_OUTPUT_DIR)
    print("Train dataset path:", sft_train_output_path)
    print("Eval dataset path:", sft_eval_output_path)
    print("New SFT training dataset size:", len(all_train))
    print("New SFT eval dataset size:", len(sft_eval))

if __name__ == "__main__":
    main()