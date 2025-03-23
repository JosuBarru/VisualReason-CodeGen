import os
import re
import random
import pandas as pd
import matplotlib.pyplot as plt

# -------------- CONFIGS --------------
input_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/all/train"
output_folder = "/sorgin1/users/jbarrutia006/viper/PrefDatasets"

accuracy_col = "accuracy"      # 1 means correct, !=1 means incorrect
sample_id_col = "sample_id"
code_col = "code"

# -------------- HELPER FUNCTIONS --------------

def remove_function_header(code_str):
    """
    Removes lines that match the function header pattern, e.g.:
      def execute_command_<sample_id>(...):
    """
    pattern = r"^def\s+execute_command_[^(]+\([^)]*\):\n"
    return re.sub(pattern, "", str(code_str), flags=re.MULTILINE)

def get_csv_files(directory):
    """Return the list of CSV filenames in the given directory."""
    return [f for f in os.listdir(directory) if f.endswith('.csv')]

def select_multiple_csv_files(csv_files):
    """Interactively select multiple CSV files from the list."""
    selected_files = []
    print("\nLista de archivos disponibles:")
    for idx, file in enumerate(csv_files):
        print(f"{idx}: {file}")

    while True:
        user_input = input("Selecciona el índice del archivo CSV (o escribe 'fin' para terminar): ")
        if user_input.lower() == 'fin':
            break
        try:
            idx = int(user_input)
            if 0 <= idx < len(csv_files):
                chosen = csv_files[idx]
                if chosen not in selected_files:
                    selected_files.append(chosen)
                else:
                    print("Ese archivo ya está seleccionado.")
            else:
                print("Índice inválido, intenta de nuevo.")
        except ValueError:
            print("Entrada no válida. Por favor, introduce un número o 'fin'.")
    return selected_files

def extract_model_name(filename):
    """
    Given a CSV file name (without path), return the model name.
    For example: "gpt4-single.csv" -> "gpt4-single"
    Adjust if you need more advanced parsing.
    """
    base = os.path.splitext(filename)[0]
    return base

def load_csvs(csv_filenames, directory):
    """
    Loads each selected CSV, infers the model name from the filename,
    appends it as a 'model' column, concatenates into one DataFrame.
    """
    df_list = []
    for filename in csv_filenames:
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            print(f"[WARNING] File not found: {path}")
            continue

        df = pd.read_csv(path, engine="python")
        # Remove last line if it's a footer
        if len(df) > 0:
            df = df.iloc[:-1]

        # Filter numeric sample_id
        df = df[df[sample_id_col].apply(lambda x: str(x).isnumeric())]

        # Add a 'model' column from the filename
        df["model"] = extract_model_name(filename)

        df_list.append(df)

    if not df_list:
        return pd.DataFrame()  # Return empty if no files loaded
    return pd.concat(df_list, ignore_index=True)

def select_valid_sample_ids(df):
    """
    For each sample_id, we need at least:
      - One 'correct' row (accuracy=1)
      - One 'incorrect' row with valid code
    Returns a list of valid sample_ids.
    """
    valid_ids = []
    grouped = df.groupby(sample_id_col)
    for sid, group in grouped:
        correct_rows = group[group[accuracy_col] == 1]
        incorrect_rows = group[group[accuracy_col] != 1]

        # Filter for non-empty code in the incorrect rows
        valid_incorrect = incorrect_rows[
            incorrect_rows[code_col].notna()
            & (incorrect_rows[code_col].apply(remove_function_header).str.strip() != "")
            & (incorrect_rows[code_col].apply(remove_function_header).str.strip().str.lower() != "nan")
        ]

        if not correct_rows.empty and not valid_incorrect.empty:
            valid_ids.append(sid)

    return valid_ids

def create_train_dev_ids(valid_ids):
    """
    Reserve first 1000 for dev, rest for train (no shuffle),
    matching your original logic.
    """
    if len(valid_ids) < 1000:
        print("[INFO] Fewer than 1000 valid sample_ids found. All go to Dev.")
        return [], valid_ids
    else:
        return valid_ids[1000:], valid_ids[:1000]

# --------------------- DPO-LIKE PAIR CREATION ---------------------
def create_pairs_for_ids(df, sample_ids, approach='single'):
    """
    Create preference pairs from the given DataFrame for specified sample_ids.
    Each pair is a dict with:
       prompt, chosen, rejected,
       chosen_model, rejected_model
    (We add the model info so we can analyze correct vs. incorrect per model later.)

    'single' => randomly pick exactly 1 correct + 1 incorrect for each sample_id.
    'all'    => produce every possible correct/incorrect combination for each sample_id.
    """
    pairs = []
    grouped = df.groupby(sample_id_col)

    for sid in sample_ids:
        group = grouped.get_group(sid)

        correct_rows = group[group[accuracy_col] == 1]
        incorrect_rows = group[group[accuracy_col] != 1]

        if correct_rows.empty or incorrect_rows.empty:
            continue
        
        # For the rejected code, try to select a code that is not NaN or empty.
        valid_incorrect_rows = incorrect_rows[incorrect_rows['code'].notna() & 
            (incorrect_rows['code'].apply(remove_function_header).str.strip() != "") & 
            (incorrect_rows['code'].apply(remove_function_header).str.strip().str.lower() != "nan")]
        

        if approach == 'single':
            # Pick 1 random correct + 1 random incorrect
            chosen_row = correct_rows.sample(n=1).iloc[0]
            rejected_row = valid_incorrect_rows.sample(n=1).iloc[0]
            pairs.append({
                'prompt': chosen_row['query'],
                'chosen': remove_function_header(chosen_row['code']),
                'rejected': remove_function_header(rejected_row['code']),
                'chosen_model': chosen_row['model'],
                'rejected_model': rejected_row['model']
            })
        else:  # approach == 'all'
            # Every possible correct vs. every valid incorrect
            for _, cr in correct_rows.iterrows():
                if not valid_incorrect_rows.empty:
                    target_incorrect = valid_incorrect_rows
                else:
                    continue
                for _, ir in target_incorrect.iterrows():
                    pairs.append({
                        'prompt': cr['query'],
                        'chosen': remove_function_header(cr['code']),
                        'rejected': remove_function_header(ir['code']),
                        'chosen_model': cr['model'],
                        'rejected_model': ir['model']
                    })

    return pairs

def flatten_pairs_to_df(pairs, split_name):
    """
    Convert the list of pair dicts into a DataFrame where
    each code snippet (correct or incorrect) is a separate row.
    That way we can easily group by model & accuracy and plot.

    We'll return columns: model, accuracy, split_name
    - chosen => accuracy=1
    - rejected => accuracy=0
    """
    rows = []
    for p in pairs:
        # chosen => correct
        rows.append({
            'model': p['chosen_model'],
            'accuracy': 1,
            'split': split_name
        })
        # rejected => incorrect
        rows.append({
            'model': p['rejected_model'],
            'accuracy': 0,
            'split': split_name
        })
    return pd.DataFrame(rows)

# --------------------- PLOTTING ---------------------

def plot_correct_incorrect_bar(df, split_name, approach, output_filename):
    """
    Plot correct vs. incorrect for each model in df. 
    'accuracy' is 1 => correct, 0 => incorrect.
    """
    summary = df.groupby("model")["accuracy"].value_counts().unstack(fill_value=0)

    # If columns are (0,1), rename them for clarity
    if 0 in summary.columns:
        summary.rename(columns={0: "Incorrect"}, inplace=True)
    if 1 in summary.columns:
        summary.rename(columns={1: "Correct"}, inplace=True)

    # Ensure columns exist and in a consistent order
    for col in ["Correct", "Incorrect"]:
        if col not in summary.columns:
            summary[col] = 0
    summary = summary[["Correct", "Incorrect"]]

    ax = summary.plot(kind="bar", figsize=(10, 5), width=0.7)
    plt.title(f"Correct vs Incorrect by Model - {split_name} ({approach.capitalize()} approach)")
    plt.xlabel("Model")
    plt.ylabel("Number of Rows")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Result")
    plt.tight_layout()

    save_path = os.path.join(output_folder, output_filename)
    plt.savefig(save_path)
    plt.show()
    print(f"[INFO] Plot saved to {save_path}")

# -------------- MAIN SCRIPT --------------

def main():
    # Step 1: Get CSV files from the folder
    csv_files = get_csv_files(input_folder)
    if not csv_files:
        print("[ERROR] No CSV files found in:", input_folder)
        return

    # Step 2: Let the user pick from the CSV list
    selected_files = select_multiple_csv_files(csv_files)
    if not selected_files:
        print("[INFO] No CSVs selected. Exiting.")
        return

    # Step 3: Ask user about approach for the DPO pairs
    approach = input("Selecciona el enfoque ('single' para un par por instancia, 'all' para todos los pares): ").strip().lower()
    if approach not in ["single", "all"]:
        print("[WARNING] Invalid approach entered. Using 'single'.")
        approach = "single"

    # Step 4: Load chosen CSVs into one DataFrame
    df = load_csvs(selected_files, input_folder)
    if df.empty:
        print("[ERROR] Combined DataFrame is empty after loading.")
        return

    # Step 5: Identify valid sample_ids
    valid_ids = select_valid_sample_ids(df)
    print(f"[INFO] Found {len(valid_ids)} valid sample IDs in total.")

    # Step 6: Split into Train & Dev sets
    train_ids, dev_ids = create_train_dev_ids(valid_ids)
    print(f"[INFO] Train IDs: {len(train_ids)}   Dev IDs: {len(dev_ids)}")

    # Step 7: Create pairs for each split
    train_df = df[df[sample_id_col].isin(train_ids)].copy()
    dev_df   = df[df[sample_id_col].isin(dev_ids)].copy()

    train_pairs = create_pairs_for_ids(train_df, train_ids, approach=approach)
    dev_pairs   = create_pairs_for_ids(dev_df,   dev_ids,   approach=approach)
    print(f"[INFO] Created {len(train_pairs)} pairs for Train, {len(dev_pairs)} for Dev.")

    # Flatten the pairs into row-level data for plotting
    train_plot_df = flatten_pairs_to_df(train_pairs, split_name="Train")
    dev_plot_df   = flatten_pairs_to_df(dev_pairs,   split_name="Dev")

    # Step 8: Plot correct vs. incorrect counts by model for each split
    if not train_plot_df.empty:
        plot_correct_incorrect_bar(
            train_plot_df,
            split_name="Train",
            approach=approach,
            output_filename=f"bar_{approach}_train.png"
        )
    else:
        print("[INFO] No train pairs to plot.")

    if not dev_plot_df.empty:
        plot_correct_incorrect_bar(
            dev_plot_df,
            split_name="Dev",
            approach=approach,
            output_filename=f"bar_{approach}_dev.png"
        )
    else:
        print("[INFO] No dev pairs to plot.")

if __name__ == "__main__":
    main()
