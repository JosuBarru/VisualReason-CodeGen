import pandas as pd
import os

file_name = "train/1codex_results_deepSeekLlama8b-post.csv"  # Cambia esto con el nombre real del archivo
input_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/all/"
output_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/metrics/"
input_file = os.path.join(input_folder, os.path.basename(file_name))
output_file = os.path.join(output_folder, os.path.basename(file_name))


df = pd.read_csv(input_file)

# Contar los ejemplos correctos
correctos = df[df['accuracy'] == 1].shape[0]

# Contar las instancias que contienen "Error" en la columna 'Answer'
errores = df[df['Answer'].str.startswith('Error', na=False)].shape[0]

# Contar el resto de las instancias
resto = df.shape[0] - correctos - errores

# Crear un DataFrame con los resultados
results_df = pd.DataFrame({
    "Errores Sintácticos o de Ejecución": [errores],
    "Errores Semánticos o de Inferencia": [resto],
    "Correctos": [correctos]
})

# Guardar el DataFrame en un archivo CSV
results_df.to_csv(output_file, index=False)

print(f"Resultados guardados en: {output_file}")
