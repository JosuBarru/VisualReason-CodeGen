import os
import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset(dataset_hf, title_suffix, approach, output_folder_path):
    """
    Visualizes the dataset by printing a sample and plotting the distribution
    of the code lengths (number of characters) for the 'chosen' and 'rejected' fields,
    using uniform bins with increased granularity and a customized x-axis range.

    Parameters:
      - dataset_hf: The Hugging Face Dataset object.
      - title_suffix: A string to append to the plot title (e.g., "(Training)").
      - approach: A string (e.g., 'single', 'all') used for naming the output plot file.
      - output_folder_path: The folder where the plot will be saved.
    """
    # Convert Hugging Face Dataset to pandas DataFrame
    df = dataset_hf.to_pandas()

    print(f"\nSample of the dataset {title_suffix}:")
    print(df.head())

    if 'chosen' not in df.columns or 'rejected' not in df.columns:
        print("Error: The dataset must contain 'chosen' and 'rejected' columns.")
        return

    df['chosen_length'] = df['chosen'].astype(str).apply(len)
    df['rejected_length'] = df['rejected'].astype(str).apply(len)

    if df['chosen_length'].empty and df['rejected_length'].empty:
        print("Warning: 'chosen' and 'rejected' columns appear to be empty or have no length.")
        # Fallback for min/max if both are empty to avoid errors, though plot will be empty.
        min_val, max_val = 0.0, 1.0 # Default small range
        # No plotting if no data
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, "No length data to plot.", ha='center', va='center')
        plt.title(f'Distribution of Code Lengths in the Dataset {title_suffix} (No Data)')
         # Ensure output folder exists
        os.makedirs(output_folder_path, exist_ok=True)
        safe_title_suffix = "".join(c if c.isalnum() else "_" for c in title_suffix)
        safe_approach = "".join(c if c.isalnum() else "_" for c in approach)
        plot_filename = f"plot_{safe_approach}_{safe_title_suffix}_nodata.svg"
        plot_filepath = os.path.join(output_folder_path, plot_filename)
        try:
            plt.savefig(plot_filepath, format='svg')
            print(f"Empty plot saved to: {plot_filepath}")
        except Exception as e:
            print(f"Error saving empty plot: {e}")
        plt.show()
        return

    elif df['chosen_length'].empty:
        print("Warning: 'chosen' column appears to be empty or has no length.")
        if df['rejected_length'].empty: # Should be caught above, but as a safeguard
             min_val, max_val = 0.0, 1.0
        else:
            min_val = df['rejected_length'].min()
            max_val = df['rejected_length'].max()
    elif df['rejected_length'].empty:
        print("Warning: 'rejected' column appears to be empty or has no length.")
        min_val = df['chosen_length'].min()
        max_val = df['chosen_length'].max()
    else:
        min_val = min(df['chosen_length'].min(), df['rejected_length'].min())
        max_val = max(df['chosen_length'].max(), df['rejected_length'].max())

    if pd.isna(min_val) or pd.isna(max_val):
        print("Error: Could not determine min/max values for lengths. Check data.")
        return

    preferred_display_max_x = 2500.0 
                                     
    num_bins = 40                    

    bin_end_val = float(max_val)

    # If the actual max value is greater than our preferred display max,
    # cap the binning range at the preferred display max.
    if max_val > preferred_display_max_x:
        bin_end_val = preferred_display_max_x

    # If the above capping results in bin_end_val being less than min_val
    # (e.g., preferred_display_max_x is set lower than the minimum data value),
    # then revert to using actual max_val for the bin end to make a sensible plot.
    if bin_end_val < float(min_val):
        bin_end_val = float(max_val)
    
    # Ensure min_val and bin_end_val are floats for linspace
    f_min_val = float(min_val)

    # Define bins
    # If min_val and bin_end_val are the same (or very close), create a small range for bins.
    if abs(f_min_val - bin_end_val) < 1e-6:
        bins = np.linspace(f_min_val - 1.0, bin_end_val + 1.0, num_bins + 1)
        current_xlim_right = bin_end_val + 1.0
        current_xlim_left = f_min_val -1.0
    else:
        bins = np.linspace(f_min_val, bin_end_val, num_bins + 1)
        current_xlim_right = bin_end_val
        current_xlim_left = f_min_val
    # --- End X-axis customization ---

    plt.figure(figsize=(12, 6))

    if not df['chosen_length'].empty:
        plt.hist(df['chosen_length'], bins=bins, alpha=0.6, label='Chosen Code Length', rwidth=0.85)
    if not df['rejected_length'].empty:
        plt.hist(df['rejected_length'], bins=bins, alpha=0.6, label='Rejected Code Length', rwidth=0.85)

    plt.xlabel('Code Length (number of characters)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Code Lengths in the training dataset single-pair approach')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)

    # Set x-axis limits for display
    plt.xlim(left=current_xlim_left, right=current_xlim_right)

    os.makedirs(output_folder_path, exist_ok=True)
    safe_title_suffix = "".join(c if c.isalnum() else "_" for c in title_suffix)
    safe_approach = "".join(c if c.isalnum() else "_" for c in approach)

    plot_filename = f"plot_{safe_approach}_{safe_title_suffix}.svg" # Changed to SVG
    plot_filepath = os.path.join(output_folder_path, plot_filename)

    try:
        plt.savefig(plot_filepath, format='svg') # Explicitly set format to svg
        print(f"Plot saved to: {plot_filepath}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.show()

# The main function remains the same as your corrected version for loading the dataset
def main():
    print("--- Dataset Visualization Script ---")

   
    dataset_dir_path = "syntData/PrefDatasets/dpo_dataset_single_train.arrow"
    output_folder = "syntData/PrefDatasets"
    title_suffix = "Training"
    approach = "single"

    print(f"\nLoading dataset from directory: {dataset_dir_path}...")
    try:
        loaded_dataset = Dataset.load_from_disk(dataset_dir_path)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Visualizing dataset...")
    visualize_dataset(loaded_dataset, title_suffix, approach, output_folder)
    print("\n--- Visualization Complete ---")

if __name__ == "__main__":
    main()