from datasets import load_from_disk
import pandas as pd

# Load the dataset from disk
dataset_path = "/sorgin1/users/jbarrutia006/viper/syntData/SFTDatasets/sft_dataset_train.arrow"
dataset = load_from_disk(dataset_path)

# Convert to pandas DataFrame
df = dataset.to_pandas()

# Take the first 100 instances
df_sample = df.head(100)

# Optionally, truncate long outputs for better readability
def truncate_text(text, max_length=500):
    return text if len(text) <= max_length else text[:max_length] + '...'

df_sample['output_truncated'] = df_sample['output'].apply(lambda x: truncate_text(x))
df_sample['prompt_truncated'] = df_sample['prompt'].apply(lambda x: truncate_text(x))

# Select relevant columns for display
df_display = df_sample[['prompt_truncated', 'output_truncated', 'model_name']]

# Display in Jupyter or export to HTML
print(df_display)

# Optionally export to HTML for better visual browsing
df_display.to_html("/sorgin1/users/jbarrutia006/viper/syntData/SFTDatasets/first_100_instances.html", index=False)

print("Visualización exportada como HTML.")
