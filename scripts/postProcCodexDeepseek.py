import csv
import re
import sys, os
script_dir = os.path.abspath('/sorgin1/users/jbarrutia006/viper/results/gqa/codex_results/train/')
sys.path.append(script_dir)
os.chdir(script_dir)

def clean_code(code):
    
    parts = code.split("def execute_command")
    if len(parts) < 2:
        return ""
    
    # Get the last part (without adding back the header)
    function_code = parts[-1]
    
    # Remove the header: find the first colon and take everything after it.
    colon_index = function_code.find(":")
    if colon_index == -1:
        return ""
    function_body = function_code[colon_index + 1:]
    
    # Crop the function body at the first occurrence of a double newline or markdown code fence.
    idx_newline = function_body.find("\n\n")
    idx_code_fence = function_body.find("```")
    
    # If a marker is not found, set its index to the length of the function body.
    if idx_newline == -1:
        idx_newline = len(function_body)
    if idx_code_fence == -1:
        idx_code_fence = len(function_body)
    
    # Determine the earliest marker index to crop the text.
    cut_index = min(idx_newline, idx_code_fence)
    
    return function_body[:cut_index]

def process_csv(input_file, output_file):
    with open(input_file, newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_MINIMAL)
                
        for row in reader:
            if len(row) < 3:
                writer.writerow(row)
                continue
            row[2] = clean_code(row[2])
            writer.writerow(row)

# Uso
process_csv("deepSeekLlama8b.csv", "deepSeekLlama8b-post.csv")
