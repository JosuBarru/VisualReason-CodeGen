#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from typing import List, Dict

# Constants for input and output folders
INPUT_FOLDER = "/sorgin1/users/jbarrutia006/viper/results/gqa/all/train"
OUTPUT_FOLDER = "/sorgin1/users/jbarrutia006/viper/syntData/SFTDatasets"

def get_csv_files(directory: str) -> List[str]:
    """
    Return a list of CSV filenames in the given directory.
    """
    return [f for f in os.listdir(directory) if f.endswith('.csv')]

def select_multiple_csv_files(csv_files: List[str]) -> List[str]:
    """
    Interactively select multiple CSV files from the list.
    """
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
            if idx < 0 or idx >= len(csv_files):
                print("Índice inválido, intenta de nuevo.")
            else:
                selected_files.append(csv_files[idx])
        except ValueError:
            print("Entrada no válida. Por favor, introduce un número o 'fin'.")
    return selected_files

def parse_model_name(file_name: str) -> str:
    """
    Given a filename like 'eval_modelName.csv',
    return just 'modelName' via regex.
    If it doesn't match the pattern, return the full filename as fallback.
    """
    match = re.match(r'^eval_(.*)\.csv$', file_name)
    if match:
        return match.group(1)
    return file_name

def load_csvs(file_paths: List[str]) -> pd.DataFrame:
    """
    Load multiple CSV files into a single pandas DataFrame.
    Adds a column 'model_name' extracted from each file's name.
    
    Assumes each CSV has a header and an extra footer line that should be removed.
    """
    df_list = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, engine='python')
        # Remove the last line (footer) if present.
        if len(df) > 0:
            df = df.iloc[:-1]
        # Extract model name and add as a constant column in this DF
        base_name = os.path.basename(file_path)  
        df["model_name"] = parse_model_name(base_name)
        
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    # Filter rows with a numeric sample_id
    combined_df = combined_df[combined_df['sample_id'].apply(lambda x: str(x).isnumeric())]
    return combined_df

def remove_function_header(code_str: str) -> str:
    """
    Remove function header lines from the given code string.
    Function headers (e.g. "def execute_command_<sample_id>(...):") are removed.
    """
    pattern = r"^def\s+execute_command_[^(]+\([^)]*\):\n"
    return re.sub(pattern, "", code_str, flags=re.MULTILINE)

def create_sft_instances(df: pd.DataFrame, sample_ids: List[str]) -> List[Dict[str, str]]:
    """
    Create supervised fine-tuning instances for the given sample_ids.
    For each sample_id, one correct (accuracy == 1) example is selected.
    
    Returns a list of dictionaries with keys:
      - 'prompt': the query text.
      - 'output': the cleaned correct code.
      - 'model_name': the model from which the correct code came.
    """
    instances = []
    grouped = df.groupby('sample_id')
    for sample_id in sample_ids:
        group = grouped.get_group(sample_id)
        # Filter for correct rows and further ensure the code is valid
        correct_rows = group[group['accuracy'] == 1]
        if correct_rows.empty:
            continue
        # Randomly select one correct example per instance
        chosen_row = correct_rows.sample(n=1).iloc[0]
        instances.append({
            'prompt': chosen_row['query'],
            'output': remove_function_header(chosen_row['code']),
            'model_name': chosen_row['model_name']
        })
    return instances


def main() -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

    # Determine valid sample_ids with at least one correct example
    valid_sample_ids = []
    grouped = df.groupby('sample_id')
    for sample_id in df['sample_id'].unique():
        group = grouped.get_group(sample_id)
        correct_rows = group[group['accuracy'] == 1]
        if not correct_rows.empty:
            valid_sample_ids.append(sample_id)

    print(f"Found {len(valid_sample_ids)} valid instances for SFT.")

    # Reserve 1000 instances for the development partition (if available)
    if len(valid_sample_ids) < 1000:
        raise ValueError("Not enough valid instances for SFT. Need at least 1000.")
    else:
        dev_ids = valid_sample_ids[:1000]
        train_ids = valid_sample_ids[1000:]

    # Create SFT instances for training and development partitions
    dev_instances = create_sft_instances(df, dev_ids)
    train_instances = create_sft_instances(df, train_ids)

    # Create HuggingFace Datasets for SFT (with 'prompt', 'output', 'model_name')
    dataset_train = Dataset.from_dict({
        'prompt': [inst['prompt'] for inst in train_instances],
        'output': [inst['output'] for inst in train_instances],
        'model_name': [inst['model_name'] for inst in train_instances]
    })

    dataset_dev = Dataset.from_dict({
        'prompt': [inst['prompt'] for inst in dev_instances],
        'output': [inst['output'] for inst in dev_instances],
        'model_name': [inst['model_name'] for inst in dev_instances]
    })

    output_train_path = os.path.join(OUTPUT_FOLDER, "sft_dataset_train.arrow")
    output_dev_path = os.path.join(OUTPUT_FOLDER, "sft_dataset_dev.arrow")

    dataset_train.save_to_disk(output_train_path)
    dataset_dev.save_to_disk(output_dev_path)

    print(f"\nDataset de entrenamiento guardado en: {output_train_path}")
    print(f"Dataset de desarrollo guardado en: {output_dev_path}")
    print(f"Número de instancias en el dataset de entrenamiento: {len(dataset_train)}")
    print(f"Número de instancias en el dataset de desarrollo: {len(dataset_dev)}")

if __name__ == "__main__":
    main()
