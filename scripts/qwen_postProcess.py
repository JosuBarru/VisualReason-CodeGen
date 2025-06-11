import pandas as pd
import csv
import re
import sys, os
script_dir = os.path.abspath('/sorgin1/users/jbarrutia006/viper/results/gqa/codex_results/testdev/')
sys.path.append(script_dir)
os.chdir(script_dir)

# Load the CSV file
input_csv = "qwen25_inst___06-10_17-02.csv"
output_csv = "qwen25_inst.csv"

# Load the CSV into a DataFrame
df = pd.read_csv(input_csv)

def should_remove(code):
    if not isinstance(code, str):
        return False
    keywords = ["for", "range"]
    coordinates = [".height", ".lower", ".left", ".right"]
    return all(k in code for k in keywords) and any(c in code for c in coordinates)

# Apply filter to the 'generated_code' column
df['generated_code'] = df['generated_code'].apply(lambda x: "wrong code" if should_remove(x) else x)

# Save the cleaned CSV
df.to_csv(output_csv, index=False, quoting=1)

print(f"Filtered CSV saved to {output_csv}")