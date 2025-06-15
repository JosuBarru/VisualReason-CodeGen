import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Things to change
metrics_dir = "/sorgin1/users/jbarrutia006/viper/results/gqa/all/testdev"
save_dir = "/sorgin1/users/jbarrutia006/viper/results/gqa/metrics/testdev"
#################


def get_csv_files(directory):
    """Returns a list of all CSV files in the given directory."""
    return [f for f in os.listdir(directory) if f.endswith('.csv')]

def select_csv_files(csv_files):
    """Prompts the user to select two CSV files from a list."""
    print("List of available files:")
    for idx, file in enumerate(csv_files):
        print(f"{idx}: {file}")
    
    idx1 = int(input("Select the index of the first CSV file: "))
    idx2 = int(input("Select the index of the second CSV file: "))
    
    return csv_files[idx1], csv_files[idx2]

def categorize_instance(answer, accuracy):
    """Categorizes an instance based on its answer and accuracy."""
    if accuracy == 1:
        return 3  # Correct
    elif isinstance(answer, str):
        if answer.startswith('Code Error'):
            return 0  # Code Error
        elif answer.startswith('Execution Error'):
            return 1  # Execution Error
    return 2  # Inference/Semantic Error

def compute_confusion_matrix(file1, file2):
    """Computes a confusion matrix between two result files."""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    matrix = np.zeros((4, 4), dtype=int)
    
    for _, row1 in df1.iterrows():
        sample_id = row1['sample_id']
        row2 = df2[df2['sample_id'] == sample_id]
        
        if not row2.empty:
            row2 = row2.iloc[0]
            idx1 = categorize_instance(row1['Answer'], row1['accuracy'])
            idx2 = categorize_instance(row2['Answer'], row2['accuracy'])
            matrix[idx1][idx2] += 1
    
    return matrix

def plot_confusion_matrix(matrix, model1_name, model2_name):
    """Plots and saves a confusion matrix."""
    labels = ['Code Errors', 'Execution Errors', 'Semantic or Inference Errors', 'Correct']
    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
    
    conf_mat_dir = os.path.join(save_dir, "conf_mat")
    os.makedirs(conf_mat_dir, exist_ok=True)
    
    plot_filename = f"{model1_name}_vs_{model2_name}.png"
    plot_path = os.path.join(conf_mat_dir, plot_filename)
    
    plt.figure(figsize=(12, 10))  # Increase size
    sns.heatmap(df_matrix, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
    
    plt.xlabel(f"Model: {model2_name}", fontsize=14)
    plt.ylabel(f"Model: {model1_name}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.title("Comparative Confusion Matrix", fontsize=16)
    plt.tight_layout()  # Automatically adjusts to prevent overlaps
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Confusion matrix saved at: {plot_path}")

def main():
    """Main function to run the script."""
    csv_files = get_csv_files(metrics_dir)
    file1, file2 = select_csv_files(csv_files)
    file1_path = os.path.join(metrics_dir, file1)
    file2_path = os.path.join(metrics_dir, file2)
    
    confusion_matrix = compute_confusion_matrix(file1_path, file2_path)
    
    plot_confusion_matrix(confusion_matrix, file1, file2)


if __name__ == "__main__":
    main()