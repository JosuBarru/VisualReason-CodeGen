import csv
import re
import sys, os
script_dir = os.path.abspath('/sorgin1/users/jbarrutia006/viper/results/gqa/codex_results/train/')
sys.path.append(script_dir)
os.chdir(script_dir)

def clean_code(code):
    """Elimina código a partir de 'def' y borra comentarios previos."""
    lines = code.split("\n")
    
    # Buscar línea con 'def' y su posible comentario anterior
    for i, line in enumerate(lines):
        if line.strip().startswith("def"):
            # Si la línea anterior es un comentario, eliminarla también
            if i > 0 and lines[i - 1].strip().startswith("#"):
                cleaned= "\n".join(lines[:i - 1]).strip()
            else:
                cleaned= "\n".join(lines[:i]).strip()
            #cleaned += '"'
            return cleaned
    
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
process_csv("codex_results_deepSeekLlama8b.csv", "codex_results_deepSeekLlama8b-post.csv")
