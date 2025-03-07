import os
import pandas as pd
from my_datasets import load_from_disk
from pprint import pprint

def filter_single_line_rejected(dataset_split, split_name):
    """
    Filters rows in the given dataset split where the 'rejected' field 
    contains one or fewer lines and prints the count and details.
    
    Parameters:
      - dataset_split: a HuggingFace Dataset split (train or dev).
      - split_name: a string indicating the name of the split.
    """
    # Convert to a DataFrame for easier processing.
    df = pd.DataFrame(dataset_split)
    
    # Replace NaN with an empty string to avoid errors.
    df['rejected'] = df['rejected'].fillna("")
    
    # Count lines using splitlines() for the 'rejected' field.
    df['line_count'] = df['rejected'].apply(lambda x: len(str(x).splitlines()))
    
    # Filter for rows where the 'rejected' code is one line or less.
    filtered_df = df[df['line_count'] <= 1]
    
    count = len(filtered_df)
    print(f"Split '{split_name}': Found {count} rows with 'rejected' code having one or fewer lines.\n")
    
    # Convert the filtered records to a list of dictionaries and print them.
    records = filtered_df.to_dict(orient='records')
    pprint(records)

def main():
    # Use the same dataset path as in your previous script.
    dataset_path = "/sorgin1/users/jbarrutia006/viper/results/gqa/dpo_dataset/dpo_dataset_single.arrow"
    
    if not os.path.exists(dataset_path):
        print(f"The dataset path '{dataset_path}' does not exist.")
        return

    # Load the dataset (assumed to be a DatasetDict with 'train' and 'dev' splits).
    dataset = load_from_disk(dataset_path)
    
    if 'train' not in dataset or 'dev' not in dataset:
        print("The loaded dataset must contain both 'train' and 'dev' splits.")
        return
    
    print("=== Analyzing 'train' Split ===")
    filter_single_line_rejected(dataset, 'train')
    
if __name__ == "__main__":
    main()
