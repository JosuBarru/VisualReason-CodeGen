import os
import glob
from datasets import load_from_disk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_dir = "/sorgin1/users/jbarrutia006/viper/syntData/PrefDatasets" 

dataset_paths = [
    d for d in glob.glob(os.path.join(data_dir, '**', '*.arrow'), recursive=True)
    if os.path.isdir(d)
]

if not dataset_paths:
    print("No HuggingFace datasets (.arrow folders) found in the directory at the expected depth!")
    exit()

# Display the available datasets
print("Available HuggingFace datasets:")
for idx, path in enumerate(dataset_paths, 1):
    print(f"{idx}: {path}")

# Prompt the user to select a dataset by number
try:
    choice = int(input("Enter the number of the dataset you want to analyze: ")) - 1
    selected_path = dataset_paths[choice]
except (ValueError, IndexError):
    print("Invalid selection.")
    exit()

print(f"Selected dataset: {selected_path}")

# Load the dataset using HuggingFace's load_from_disk
dataset = load_from_disk(selected_path)
# If the loaded object is a DatasetDict, select a split (e.g., 'train') or the first available one
if hasattr(dataset, "keys"):
    if "train" in dataset:
        dataset = dataset["train"]
    else:
        dataset = list(dataset.values())[0]

# Convert the dataset to a pandas DataFrame
df = dataset.to_pandas()

# Check for required columns
expected_columns = {'model', 'rejected_model'}
if not expected_columns.issubset(df.columns):
    print(f"Dataset is missing one or more required columns: {expected_columns}")
    exit()

# Count occurrences for correct ('model') and rejected ('rejected_model') codes
correct_counts = df['model'].value_counts()
rejected_counts = df['rejected_model'].value_counts()

# Get the union of models from both counts
models = sorted(set(correct_counts.index) | set(rejected_counts.index))
correct_values = [correct_counts.get(model, 0) for model in models]
rejected_values = [rejected_counts.get(model, 0) for model in models]

# Create a grouped bar plot
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars_correct = ax.bar(x - width/2, correct_values, width, label='Correct Code')
bars_rejected = ax.bar(x + width/2, rejected_values, width, label='Rejected Code')

ax.set_ylabel('Number of Code Instances')
ax.set_title('Distribution of Correct and Rejected Codes by Model')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha="right")
ax.legend()

# Function to add labels above the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars_correct)
add_labels(bars_rejected)
plt.tight_layout()

# Determine the base name and output directory
base_name = os.path.basename(selected_path)
# Save the plot in the parent directory of the selected dataset folder, not inside it.
output_directory = os.path.dirname(selected_path)
output_filename = os.path.join(output_directory, f"{base_name}.svg")

plt.savefig(output_filename, format="svg")
print(f"Plot saved as {output_filename}")