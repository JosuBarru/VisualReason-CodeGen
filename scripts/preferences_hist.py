import os
import pandas as pd
from datasets import Dataset

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

def load_csvs(file_paths):
    """
    Load multiple CSV files into a single pandas DataFrame.
    Assumes each CSV has a header and an extra footer line that should be removed.
    """
    df_list = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, engine='python')
        # Remove the last line (footer) if present.
        if len(df) > 0:
            df = df.iloc[:-1]
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    # Optionally filter out rows with invalid sample_id (e.g., non-numeric)
    combined_df = combined_df[combined_df['sample_id'].apply(lambda x: str(x).isnumeric())]
    return combined_df

def create_pairs(df, approach='single'):
    """
    Create preference pairs for DPO training from the DataFrame.
    
    Uses:
      - sample_id as the instance identifier.
      - question as the prompt.
      - code as the generated code.
      - accuracy: 1 means correct; otherwise, it's considered an error.
    
    Parameters:
      - approach: 'single' for one pair per instance; 'all' for all possible pairs.
    
    Returns:
      - A list of dictionaries with keys: 'prompt', 'chosen', 'rejected'.
    """
    pairs = []
    grouped = df.groupby('sample_id')
    for sample_id, group in grouped:
        # Select correct and incorrect codes based on 'accuracy'
        correct_rows = group[group['accuracy'] == 1]
        incorrect_rows = group[group['accuracy'] != 1]
        # Only consider the instance if at least one correct code exists.
        if correct_rows.empty:
            continue
        
        if approach == 'single':
            chosen_row = correct_rows.iloc[0]
            if incorrect_rows.empty:
                continue  # Skip if no incorrect code is available
            rejected_row = incorrect_rows.iloc[0]
            pairs.append({
                'prompt': chosen_row['question'],
                'chosen': chosen_row['code'],
                'rejected': rejected_row['code']
            })
        elif approach == 'all':
            for _, correct_row in correct_rows.iterrows():
                for _, incorrect_row in incorrect_rows.iterrows():
                    pairs.append({
                        'prompt': correct_row['question'],
                        'chosen': correct_row['code'],
                        'rejected': incorrect_row['code']
                    })
    return pairs

def main():
    input_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/all/train"
    output_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/dpo_dataset/train"
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
    approach = input("Selecciona el enfoque ('single' para un par por instancia, 'all' para todos los pares): ").strip().lower()
    if approach not in ['single', 'all']:
        print("Enfoque no reconocido, se utilizará 'single' por defecto.")
        approach = 'single'
    
    # Create preference pairs.
    pairs = create_pairs(df, approach=approach)
    if not pairs:
        print("No se pudieron crear pares de preferencia. Revisa los datos.")
        return
    
    # Build a HuggingFace Dataset.
    dataset = Dataset.from_dict({
        'prompt': [pair['prompt'] for pair in pairs],
        'chosen': [pair['chosen'] for pair in pairs],
        'rejected': [pair['rejected'] for pair in pairs]
    })
    
    output_file = os.path.join(output_folder, f"dpo_dataset_{approach}.arrow")
    dataset.save_to_disk(output_file)
    print(f"Dataset de DPO guardado en: {output_file}")
    print(f"Número de instancias en el dataset: {len(dataset)}")

if __name__ == "__main__":
    main()
