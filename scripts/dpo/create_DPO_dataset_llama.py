import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset, load_from_disk

input_folder  = "/sorgin1/users/jbarrutia006/viper/results/gqa/all/train"
output_folder = "/sorgin1/users/jbarrutia006/viper/syntData/PrefDatasets"

# Arrow dataset whose *prompt* column is used to filter the rows we keep
arrow_match_path = (
    "/sorgin1/users/jbarrutia006/viper/syntData/PrefDatasets/"
    "dpo_dataset_single_train.arrow"
)

# Rejected completions **must** come from this model
TARGET_REJECTED_MODEL = "llama31-8B-16b___02-20_01-18"

def get_csv_files(directory):
    """Return a list of CSV filenames in the given directory."""
    return [f for f in os.listdir(directory) if f.endswith(".csv")]


def select_multiple_csv_files(csv_files):
    """Interactively select multiple CSV files from the list."""
    selected = []
    print("\nLista de archivos disponibles:")
    for idx, f in enumerate(csv_files):
        print(f"{idx}: {f}")
    while True:
        choice = input("Selecciona un índice (o 'fin'): ")
        if choice.lower() == "fin":
            break
        try:
            idx = int(choice)
            if 0 <= idx < len(csv_files):
                selected.append(csv_files[idx])
            else:
                print("Índice inválido.")
        except ValueError:
            print("Entrada no válida.")
    return selected


def parse_model_name(file_name: str) -> str:
    """Extract model name from filename `eval_<model>.csv`."""
    m = re.match(r"^eval_(.*)\.csv$", file_name)
    return m.group(1) if m else file_name


def load_csvs(file_paths):
    """
    Load multiple CSVs, tag each row with the originating model (from filename),
    drop footer rows and enforce numeric sample_id.
    """
    dfs = []
    for fp in file_paths:
        df = pd.read_csv(fp, engine="python")
        if len(df):
            df = df.iloc[:-1]                    # drop footer line
        df["model"] = parse_model_name(os.path.basename(fp))
        dfs.append(df)
    combo = pd.concat(dfs, ignore_index=True)
    return combo[combo["sample_id"].astype(str).str.isnumeric()]


def remove_function_header(code_str: str) -> str:
    """
    Strip lines like  'def execute_command_<id>(...):'  from generated code.
    """
    pattern = r"^def\s+execute_command_[^(]+\([^)]*\):\n"
    return re.sub(pattern, "", code_str, flags=re.MULTILINE)


def create_pairs_for_ids(df: pd.DataFrame, sample_ids, approach="single"):
    """
    Build preference pairs for the given `sample_ids`, **keeping only** rows
    where the rejected code comes from `TARGET_REJECTED_MODEL`.
    """
    pairs   = []
    grouped = df.groupby("sample_id")

    for sid in sample_ids:
        grp          = grouped.get_group(sid)

        # Correct (=chosen) rows
        correct_rows = grp[grp["accuracy"] == 1]

        # Incorrect (=rejected) rows from the **target model only**
        incorrect_rows = grp[
            (grp["accuracy"] != 1) & (grp["model"] == TARGET_REJECTED_MODEL)
        ]

        if correct_rows.empty or incorrect_rows.empty:
            continue  # must have both sides

        # Ensure rejected code is non‑empty after cleanup
        def _clean(row):
            return remove_function_header(row["code"]).strip()

        incorrect_rows = incorrect_rows[
            incorrect_rows["code"].notna()
            & incorrect_rows.apply(_clean, axis=1).str.lower().ne("nan")
            & incorrect_rows.apply(_clean, axis=1).ne("")
        ]
        if incorrect_rows.empty:
            continue

        if approach == "single":
            chosen   = correct_rows.sample(n=1).iloc[0]
            rejected = incorrect_rows.sample(n=1).iloc[0]
            pairs.append(
                {
                    "prompt": chosen["query"],
                    "chosen":   _clean(chosen),
                    "rejected": _clean(rejected),
                    "model":          chosen["model"],
                    "rejected_model": rejected["model"],
                }
            )
        else:  
            for _, c_row in correct_rows.iterrows():
                for _, ic_row in incorrect_rows.iterrows():
                    pairs.append(
                        {
                            "prompt": c_row["query"],
                            "chosen":   _clean(c_row),
                            "rejected": _clean(ic_row),
                            "model":          c_row["model"],
                            "rejected_model": ic_row["model"],
                        }
                    )

    return pairs


def visualize_dataset(dataset: Dataset, title_suffix: str, approach: str):
    """Optional quick inspection histogram (unchanged from original)."""
    df = pd.DataFrame(dataset)
    print(f"\nSample {title_suffix}:")
    print(df.head())

    df["chosen_len"]   = df["chosen"].str.len()
    df["rejected_len"] = df["rejected"].str.len()

    bins = np.linspace(
        0,
        max(df["chosen_len"].max(), df["rejected_len"].max()),
        41,
    )

    plt.figure(figsize=(10, 5))
    plt.hist(df["chosen_len"],   bins=bins, alpha=0.5, label="Chosen length",   rwidth=0.9)
    plt.hist(df["rejected_len"], bins=bins, alpha=0.5, label="Rejected length", rwidth=0.9)
    plt.xlabel("Characters in code snippet")
    plt.ylabel("Frequency")
    plt.title(f"Code‑length distribution {title_suffix}")
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"plot_{approach}_{title_suffix}.png"))
    plt.show()
    plt.close()


def main():
    os.makedirs(output_folder, exist_ok=True)

    # Load prompt set from Arrow for filtering
    print("Cargando prompts de referencia …")
    prompts_set = list(load_from_disk(arrow_match_path)["prompt"])
    print(f"   {len(prompts_set):,} prompts disponibles.\n")

    # Interactive CSV selection
    csv_files = get_csv_files(input_folder)
    if not csv_files:
        print("No se encontraron CSVs en el directorio.")
        return

    selected = select_multiple_csv_files(csv_files)
    if not selected:
        print("No se seleccionó ningún archivo. Saliendo.")
        return

    df = load_csvs([os.path.join(input_folder, f) for f in selected])
    approach = input("Selecciona el enfoque ('single' para un par por instancia, 'all' para todos los pares): ").strip().lower()
    if approach not in ['single', 'all']:
        print("Enfoque no reconocido, se utilizará 'single' por defecto.")
        approach = 'single'

    # Filter sample_ids
    grouped = df.groupby("sample_id")
    valid_ids = []
    for sid in df["sample_id"].unique():
        grp = grouped.get_group(sid)

        # Must have at least one correct row …
        if grp[grp["accuracy"] == 1].empty:
            continue

        # … and at least one incorrect row from TARGET_REJECTED_MODEL …
        incorrect_target = grp[
            (grp["accuracy"] != 1) & (grp["model"] == TARGET_REJECTED_MODEL)
        ]
        if incorrect_target.empty:
            continue

        # … and its query must match a prompt in the Arrow dataset
        query_val = grp.iloc[0]["query"]
        if query_val not in prompts_set:
            continue

        valid_ids.append(sid)

    print(f"{len(valid_ids)} valid instances found.")

    # Build pairs & dataset
    train_pairs = create_pairs_for_ids(df, valid_ids, approach)

    if not train_pairs:
        print("No train pairs found.")
        return

    dataset_train = Dataset.from_dict(
        {
            "prompt":         [p["prompt"]         for p in train_pairs],
            "chosen":         [p["chosen"]         for p in train_pairs],
            "rejected":       [p["rejected"]       for p in train_pairs],
            "model":          [p["model"]          for p in train_pairs],
            "rejected_model": [p["rejected_model"] for p in train_pairs],
        }
    )

    out_path = os.path.join(output_folder, "dpo_dataset_llama_train.arrow")
    dataset_train.save_to_disk(out_path)

    print(f"\nDataset saved {out_path}")
    print(f"Instances number: {len(dataset_train):,}")

    visualize_dataset(dataset_train, "(Filtro completo)", approach)


if __name__ == "__main__":
    main()