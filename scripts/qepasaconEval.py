import pandas as pd

import sys, os
script_dir = os.path.abspath('/sorgin1/users/jbarrutia006/viper/results/gqa/codex_results/train/')
sys.path.append(script_dir)
os.chdir(script_dir)

# Read the CSV file (adjust filename and header as needed)
df = pd.read_csv("Qwen257b.csv", header=None)

# Note: pandas uses 0-indexing, so the 9280th row is at index 9279.
subset = df.iloc[9279:9279+65]

# Write the selected rows to a new CSV file
subset.to_csv('selected_rows.csv', index=False, header=False)
