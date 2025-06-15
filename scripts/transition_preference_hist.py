import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm

# --- Core Functions (unchanged) ---

def get_classification_bits(answer, accuracy):
    """
    Returns a 4-bit list for the following categories:
      [Compilation Error, Runtime Error, Semantic/Inference Error, Correct]
    """
    if accuracy == 1:
        return [0, 0, 0, 1]
    elif isinstance(answer, str):
        if 'Compilation Error' in answer or 'Error Codigo' in answer:
            return [1, 0, 0, 0]
        elif 'Runtime Error' in answer or 'Error Ejecucion' in answer:
            return [0, 1, 0, 0]
    return [0, 0, 1, 0]

def process_files_for_patterns(list_of_files, directory):
    """
    Processes a list of CSV files and returns a dictionary with the
    combined classification pattern for each instance.
    """
    instance_classifications = {}
    for file_name in list_of_files:
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: The file {file_path} was not found and will be skipped.")
            continue
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            continue
        required_cols = ['sample_id', 'Answer', 'accuracy']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: The file {file_name} is missing required columns and will be skipped.")
            continue
        if not df.empty and 'TOTAL' in str(df.iloc[-1]['sample_id']):
             df = df.iloc[:-1]
        print(f"Processing {file_name}...")
        for _, row in df.iterrows():
            sample_id = row['sample_id']
            try:
                accuracy = pd.to_numeric(row['accuracy'])
            except (ValueError, TypeError):
                accuracy = 0
            bits = get_classification_bits(row['Answer'], accuracy)
            if sample_id not in instance_classifications:
                instance_classifications[sample_id] = bits
            else:
                instance_classifications[sample_id] = [max(e, n) for e, n in zip(instance_classifications[sample_id], bits)]
    return {sample_id: tuple(bits) for sample_id, bits in instance_classifications.items()}

def select_files_interactive(all_files, group_name):
    """
    Displays a list of files and prompts the user to select multiple files.
    """
    print("-" * 50)
    print(f"\nSelect the files for the '{group_name.upper()}' models")
    print("Available files:")
    for idx, file_name in enumerate(all_files):
        print(f"  [{idx}] {file_name}")
    selected_files = []
    while True:
        try:
            user_input = input(f"\nEnter the file numbers separated by commas (e.g., 0,2,5): ")
            if not user_input:
                print("No files selected. Please try again.")
                continue
            selected_indices = [int(i.strip()) for i in user_input.split(',')]
            if any(i < 0 or i >= len(all_files) for i in selected_indices):
                print("Error: One or more numbers are out of range. Please try again.")
                continue
            selected_files = [all_files[i] for i in selected_indices]
            print(f"\nYou have selected for '{group_name}':")
            for f in selected_files:
                print(f"  - {f}")
            confirm = input("Is this selection correct? (y/n): ").lower()
            if confirm == 'y':
                break
            else:
                print("Selection cancelled. Please start over.")
        except ValueError:
            print("Error: Invalid input. Please enter only numbers separated by commas.")
    return selected_files

# --- Main script ---

def main():
    # 1. Setup
    data_directory = 'results/gqa/all/testdev'
    try:
        all_csv_files = sorted([f for f in os.listdir(data_directory) if f.endswith('.csv')])
    except FileNotFoundError:
        print(f"Error: The directory '{data_directory}' does not exist.")
        return
    if not all_csv_files:
        print(f"No .csv files found in the directory '{data_directory}'.")
        return

    # 2. Interactive file selection
    original_files = select_files_interactive(all_csv_files, "Not fine-tuned")
    finetuned_files = select_files_interactive(all_csv_files, "Fine-tuned")

    # 3. Process files
    print("\n--- Processing NOT FINE-TUNED model files ---")
    original_patterns = process_files_for_patterns(original_files, data_directory)
    print("\n--- Processing FINE-TUNED model files ---")
    finetuned_patterns = process_files_for_patterns(finetuned_files, data_directory)
    
    # 4. Define pattern order
    ordered_combinations = [
        (0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1), (1, 0, 1, 1),
        (0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 0), (0, 1, 1, 0),
        (1, 1, 1, 0), (1, 0, 1, 0), (0, 1, 0, 0), (1, 1, 0, 0), (1, 0, 0, 0)
    ]
    pattern_to_index = {pattern: i for i, pattern in enumerate(ordered_combinations)}
    
    # 5. Build the transition matrix
    matrix_size = len(ordered_combinations)
    transition_matrix = [[0] * matrix_size for _ in range(matrix_size)]
    all_sample_ids = set(original_patterns.keys()) | set(finetuned_patterns.keys())
    
    if not all_sample_ids:
        print("\nError: No instances found in the selected files.")
        return

    for sample_id in all_sample_ids:
        pattern_orig = original_patterns.get(sample_id)
        pattern_ft = finetuned_patterns.get(sample_id)
        if pattern_orig is not None and pattern_ft is not None:
            orig_idx = pattern_to_index.get(pattern_orig)
            ft_idx = pattern_to_index.get(pattern_ft)
            if orig_idx is not None and ft_idx is not None:
                transition_matrix[orig_idx][ft_idx] += 1
            
    # 6. Visualize and save the matrix
    print("\nGenerating the transition matrix visualization...")
    pattern_labels = [chr(65 + i) for i in range(matrix_size)]
    df_heatmap = pd.DataFrame(transition_matrix, index=pattern_labels, columns=pattern_labels)
    
    # --- START OF VISUALIZATION REFINEMENTS ---

    # Set a formal, serif font for the thesis
    plt.rcParams['font.family'] = 'serif'
    
    fig, ax = plt.subplots(figsize=(16, 14))

    # --- NEW LOGIC: Treat 0 and 1 as the same color ---
    # Create a copy of the data for coloring ONLY. Replace 0s with 1s.
    df_for_coloring = df_heatmap.replace(0, 1)

    # The data for annotations remains the original, to show the true numbers.
    annot_data = df_heatmap

    # Use a standard LogNorm. Since the minimum value for coloring is now 1, this is safe.
    norm_scale = LogNorm()
    
    # Create the heatmap with the new coloring and original annotations
    sns.heatmap(
        df_for_coloring,    # Use the modified data for colors
        annot=annot_data,     # Use the original data for annotations
        fmt='d',
        cmap='viridis',
        linewidths=.5,
        ax=ax,
        norm=norm_scale,
        cbar_kws={'label': 'Number of Instances (Log Scale, 0 visually same as 1)'}
    )
    
    # Move the x-axis (column) ticks and label to the top
    ax.tick_params(axis='x', labeltop=True, labelbottom=False, top=True, bottom=False)
    ax.set_xlabel('Pattern in "Fine-tuned" Models (A-O)', fontsize=12, labelpad=15)
    ax.xaxis.set_label_position('top')
    
    ax.set_ylabel('Pattern in "Not fine-tuned" Models (A-O)', fontsize=12)

    plt.tight_layout()
    
    # 7. Save the output as SVG
    output_svg_file = "transition_matrix_0_as_1.svg"
    output_csv_file = "transition_matrix_data.csv"
    plt.savefig(output_svg_file, format='svg')
    df_heatmap.to_csv(output_csv_file)
    
    print(f"\nSuccess!")
    print(f"Formal SVG chart saved as: '{output_svg_file}'")
    print(f"Matrix data saved in: '{output_csv_file}'")

if __name__ == "__main__":
    main()