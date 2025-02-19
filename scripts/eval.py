import pandas as pd
import os


#Cambiar
input_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/all/train"
output_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/metrics/train"
########


def get_csv_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.csv')]

def select_csv_file(csv_files):
    print("Lista de archivos disponibles:")
    for idx, file in enumerate(csv_files):
        print(f"{idx}: {file}")
    
    idx = int(input("Selecciona el índice del archivo CSV: "))
    return csv_files[idx]

def main():
    csv_files = get_csv_files(input_folder)
    file_name = select_csv_file(csv_files)
    
    input_file = os.path.join(input_folder, file_name)
    output_file = os.path.join(output_folder, file_name)
    
    df = pd.read_csv(input_file)
    
    # Contar los ejemplos correctos
    correctos = df[df['accuracy'] == 1].shape[0]
    
    # Contar las instancias que contienen "Error" en la columna 'Answer'
    errores_cod = df[df['Answer'].str.startswith('Error Codigo', na=False)].shape[0]
    errores_ejec = df[df['Answer'].str.startswith('Error Ejecucion', na=False)].shape[0]
    
    # Contar el resto de las instancias
    resto = df.shape[0] - correctos - errores_cod - errores_ejec
    
    # Crear un DataFrame con los resultados
    results_df = pd.DataFrame({
        "Errores Codigo": [errores_cod],
        "Errores Ejecución": [errores_ejec],
        "Errores Semánticos o de Inferencia": [resto],
        "Correctos": [correctos]
    })
    
    # Guardar el DataFrame en un archivo CSV
    results_df.to_csv(output_file, index=False)
    
    print(f"Resultados guardados en: {output_file}")

if __name__ == "__main__":
    main()
