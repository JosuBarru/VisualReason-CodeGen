import csv
import re
import sys, os
script_dir = os.path.abspath('/sorgin1/users/jbarrutia006/viper/results/gqa/codex_results/train/')
sys.path.append(script_dir)
os.chdir(script_dir)

def clean_code(code):
    """Elimina código a partir de 'def' y borra comentarios previos."""
    if "while true" in code.lower():
        return ""
    
    return code

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
process_csv("Qwen257b.csv", "Qwen257b-post.csv")
