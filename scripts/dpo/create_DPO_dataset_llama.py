import os
import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
import re
import numpy as np

input_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/all/train"
output_folder = "/sorgin1/users/jbarrutia006/viper/syntData/PrefDatasets"

def get_csv_files(directory):
    """Return a list of CSV filenames in the given directory."""
    return [f for f in os.listdir(directory) if f.endswith('.csv')]

def select_multiple_csv_files(csv_files):
    """Interactively select multiple CSV files from the list."""
    selected_files = []
    print("Lista de archivos disponibles:")
    for idx, file in enumerate(csv_files):
        print(f"{idx}: {file}")
    while True:
        user_input = input("Selecciona el índice del archivo CSV (o escribe 'fin' para terminar): ")
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

def load_csvs(file_paths):
    """
    Load multiple CSV files into a single pandas DataFrame.
    Each CSV file is assigned a 'model' column derived from its file name.
    Assumes each CSV has a header and an extra footer line that should be removed.
    """
    df_list = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, engine='python')
        # Remove the last line (footer) if present.
        if len(df) > 0:
            df = df.iloc[:-1]
        # Extract model name from the file name and add it as a new column.
        file_name = os.path.basename(file_path)
        df['model'] = parse_model_name(file_name)

        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    # Optionally filter out rows with invalid sample_id (e.g., non-numeric)
    combined_df = combined_df[combined_df['sample_id'].apply(lambda x: str(x).isnumeric())]
    return combined_df

def remove_function_header(code_str):
    """
    Remove all function header lines from the generated code.
    
    The function headers are assumed to follow the pattern:
      def execute_command_<sample_id>(...):
      
    This function uses a regular expression to remove any line that starts with
    'def execute_command_' and ends with a colon.
    """
    pattern = r"^def\s+execute_command_[^(]+\([^)]*\):\n"
    return re.sub(pattern, "", code_str, flags=re.MULTILINE)

def create_pairs_for_ids(df, sample_ids):
    """
    Create preference pairs for DPO training from the DataFrame for given sample_ids.
    
    Uses:
      - sample_id as the instance identifier.
      - query as the prompt.
      - code as the generated code.
      - accuracy: 1 means correct; otherwise, it's considered an error.
      - model: the model that generated the code.
    
    Parameters:
      - sample_ids: a list of sample_ids (instances) to process.
      - approach: 'single' for one pair per instance; 'all' for all possible pairs.
    
    Returns:
      - A list of dictionaries with keys: 'prompt', 'chosen', 'rejected', 'model', 'rejected_model'.
    """
    pairs = []
    grouped = df.groupby('sample_id')
    for sample_id in sample_ids:
        group = grouped.get_group(sample_id)
        # Select correct and incorrect codes based on 'accuracy'
        correct_rows = group[group['accuracy'] == 1]
        incorrect_rows = group[group['accuracy'] != 1]
        if correct_rows.empty or incorrect_rows.empty:
            continue
        
        # For the rejected code, try to select a code that is not NaN or empty.
        valid_incorrect_rows = incorrect_rows[
            (incorrect_rows['model'] == "llama31-8B-16b___02-20_01-18") &
            incorrect_rows['code'].notna() & 
            (incorrect_rows['code'].apply(remove_function_header).str.strip() != "") & 
            (incorrect_rows['code'].apply(remove_function_header).str.strip().str.lower() != "nan")
        ]

        for _, correct_row in correct_rows.iterrows():
            if not valid_incorrect_rows.empty:
                target_incorrect = valid_incorrect_rows
            else:
                continue
            for _, incorrect_row in target_incorrect.iterrows():
                pairs.append({
                    'prompt': correct_row['query'],
                    'chosen': remove_function_header(correct_row['code']),
                    'rejected': remove_function_header(incorrect_row['code']),
                    'model': correct_row['model'],
                    'rejected_model': incorrect_row['model']
                })
    return pairs

def visualize_dataset(dataset, title_suffix):
    """
    Visualizes the dataset by printing a sample and plotting the distribution
    of the code lengths (number of characters) for the 'chosen' and 'rejected' fields,
    using uniform bins with increased granularity.
    """
    df = pd.DataFrame(dataset)
    print(f"\nSample of the dataset {title_suffix}:")
    print(df.head())
    
    # Compute lengths of code strings
    df['chosen_length'] = df['chosen'].apply(lambda x: len(str(x)))
    df['rejected_length'] = df['rejected'].apply(lambda x: len(str(x)))
    
    # Determine common uniform bins with higher granularity (e.g., 40 bins)
    min_val = min(df['chosen_length'].min(), df['rejected_length'].min())
    max_val = max(df['chosen_length'].max(), df['rejected_length'].max())
    bins = np.linspace(min_val, max_val, 41)  # 40 bins

    plt.figure(figsize=(10, 5))
    plt.hist(df['chosen_length'], bins=bins, alpha=0.5, label='Chosen Code Length', rwidth=0.9)
    plt.hist(df['rejected_length'], bins=bins, alpha=0.5, label='Rejected Code Length', rwidth=0.9)
    plt.xlabel('Code Length (number of characters)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Code Lengths in the Dataset {title_suffix}')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"plot_justLlamaErr_{title_suffix}.png"))
    plt.show()

def main():
    os.makedirs(output_folder, exist_ok=True)
    
    # List available CSV files.
    csv_files = get_csv_files(input_folder)
    if not csv_files:
        print("No se encontraron archivos CSV en el directorio.")
        return

    selected_files = select_multiple_csv_files(csv_files)
    if not selected_files:
        print("No se seleccionó ningún archivo. Saliendo.")
        return

    # Build full paths for the selected files.
    file_paths = [os.path.join(input_folder, f) for f in selected_files]
    df = load_csvs(file_paths)
    
    # Ask the user which pairing approach to use.
    print("Default approach is all for llama31 specific dataset")
    
    valid_sample_ids = []
    grouped = df.groupby('sample_id')
    for sample_id in df['sample_id'].unique():
        group = grouped.get_group(sample_id)
        correct_rows = group[group['accuracy'] == 1]
        incorrect_rows = group[group['accuracy'] != 1]
        incorrect_rows = group[
            (group['accuracy'] != 1) & 
            (group['model'] == "llama31-8B-16b___02-20_01-18") & 
            group['code'].notna() & 
            (group['code'].apply(remove_function_header).str.strip() != "") & 
            (group['code'].apply(remove_function_header).str.strip().str.lower() != "nan")
        ]
        if not correct_rows.empty and not incorrect_rows.empty:
            valid_sample_ids.append(sample_id)
    
    print(f"Found {len(valid_sample_ids)} valid instances with rejected code from 'llama31'.")    

    
    # Create preference pairs separately for development and training partitions.
    train_pairs = create_pairs_for_ids(df, valid_sample_ids)
    
    # Create HuggingFace Datasets including both model and rejected_model columns.
    dataset_train = Dataset.from_dict({
        'prompt': [pair['prompt'] for pair in train_pairs],
        'chosen': [pair['chosen'] for pair in train_pairs],
        'rejected': [pair['rejected'] for pair in train_pairs],
        'model': [pair['model'] for pair in train_pairs],
        'rejected_model': [pair['rejected_model'] for pair in train_pairs]
    })
    
    output_train = os.path.join(output_folder, f"dpo_dataset_llama_train.arrow")
    
    dataset_train.save_to_disk(output_train)
    
    print(f"\nDataset de entrenamiento guardado en: {output_train}")
    print(f"Número de instancias en el dataset de entrenamiento: {len(dataset_train)}")
    
    # Visualize both partitions.
    visualize_dataset(dataset_train, title_suffix="(Entrenamiento)")

if __name__ == "__main__":
    main()
