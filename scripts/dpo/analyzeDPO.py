import os
import pandas as pd
from datasets import load_from_disk
from pprint import pprint

def filter_single_line_rejected(dataset_split):
    """
    Filters rows in the given dataset split where the 'rejected' field 
    contains one or fewer lines and prints the count and details.
    
    Parameters:
      - dataset_split: a HuggingFace Dataset split (train or dev).
      - split_name: a string indicating the name of the split.
    """
    # Convert to a DataFrame for easier processing.
    df = pd.DataFrame(dataset_split)
    
    for i, row in df.head(50).iterrows():
        print(f"Row {i}:\nPrompt: {row['prompt']}\nChosen: {row['chosen']}\nRejected: {row['rejected']}\n")


def main():
    # Use the same dataset path as in your previous script.
    dataset_path = "/sorgin1/users/jbarrutia006/viper/PrefDatasets/dpo_dataset_single_train.arrow"
    
    if not os.path.exists(dataset_path):
        print(f"The dataset path '{dataset_path}' does not exist.")
        return

    # Load the dataset (assumed to be a DatasetDict with 'train' and 'dev' splits).
    dataset = load_from_disk(dataset_path)
    
    print("=== Analyzing ===")
    filter_single_line_rejected(dataset)
    
if __name__ == "__main__":
    main()
