import pandas as pd
import datasets
import matplotlib.pyplot as plt
import numpy as np

# --- COLOR SCHEME DEFINITION (for Plot 2 / Histograms) ---
# This remains the same to keep consistency with your thesis
color_all_csv = '#ff7f0e'      # The orange from your thesis plot
color_hf_selected = '#1f77b4' # The blue from your thesis plot
# ------------------------------------


# --- 1. Load Datasets ---

try:
    hf_dataset = datasets.load_from_disk("syntData/SFTDatasets/sft_dataset_train_experimental.arrow")
except FileNotFoundError:
    print("Error: The experimental dataset was not found.")
    print("Please make sure you have run the data creation scripts and the path is correct.")
    exit()

df_hf = hf_dataset.to_pandas()
df_orig = pd.read_csv("results/gqa/all/train/eval_llama31-8B-16b___02-20_01-18.csv")

print(f"HF dataset length: {len(df_hf)}")
print(f"Original CSV dataset length: {len(df_orig)}")


# --- 2. Prepare for Merge on 'sample_id' ---

df_hf['sample_id'] = df_hf['sample_id'].astype(str)
df_orig['sample_id'] = df_orig['sample_id'].astype(str)
df_orig_unique = df_orig.drop_duplicates(subset=['sample_id'], keep='first')
print(f"Unique sample_ids in CSV: {len(df_orig_unique)}")


# --- 3. Merge Datasets using 'sample_id' ---

merged = pd.merge(df_hf, df_orig_unique[['sample_id', 'truth_answers']], on='sample_id', how='left')
matched_count = merged['truth_answers'].notna().sum()
unmatched_count = merged['truth_answers'].isna().sum()

print(f"\nMerge Results using 'sample_id':")
print(f"Successfully matched instances: {matched_count}")
print(f"Unmatched instances: {unmatched_count}")


# --- 4. Plot Answer Frequency Distribution ---

answer_counts_all = df_orig['truth_answers'].value_counts()
answer_counts_hf = merged['truth_answers'].value_counts().dropna()

# NEW: Define stronger colors specifically for this plot
strong_color_all_csv = '#b35806'      # A darker, browner orange
strong_color_hf_selected = '#08519c' # A darker, deeper blue

plt.figure(figsize=(14, 7))

# UPDATED: Using stronger colors and no transparency (alpha=1.0)
plt.bar(
    np.arange(len(answer_counts_all)),
    answer_counts_all.values,
    label='All CSV',
    color=strong_color_all_csv,
    alpha=1.0
)
# UPDATED: Using stronger colors and no transparency (alpha=1.0)
plt.bar(
    np.arange(len(answer_counts_hf)),
    answer_counts_hf.values,
    label='HF Selected',
    color=strong_color_hf_selected,
    alpha=1.0
)

plt.yscale('log')
plt.xlabel("Answer Rank (Sorted Independently)")
plt.ylabel("Count (log scale)")
plt.title("Independently Sorted Answer Frequency Distributions")
plt.legend(title="Dataset")
plt.xticks([], [])
plt.tight_layout()
plt.savefig("answer_frequency_independently_sorted.svg")
plt.close()


# --- 5. Plot Prompt Length Distribution ---

merged['prompt_length'] = merged['prompt'].str.len()
df_orig['query_length'] = df_orig['query'].str.len()

all_lengths = np.concatenate([
    merged['prompt_length'].dropna().values,
    df_orig['query_length'].dropna().values
])
bins = np.histogram_bin_edges(all_lengths, bins=40)

plt.figure(figsize=(10, 6))

# NOTE: This plot still uses the original thesis colors for consistency
plt.hist(df_orig['query_length'], bins=bins, alpha=0.6, label='All CSV', color=color_all_csv, edgecolor='black')
plt.hist(merged['prompt_length'], bins=bins, alpha=0.7, label='HF Selected', color=color_hf_selected, edgecolor='black')

plt.yscale('log')
plt.xlabel("Prompt Length (characters)")
plt.ylabel("Count (log scale)")
plt.title("Prompt Length Distribution: Selected vs All")
plt.legend(title="Dataset")
plt.tight_layout()
plt.savefig("prompt_length_hist_overlaid.svg")
plt.close()

print("\nAnalysis complete. Plots have been saved as .svg files.")