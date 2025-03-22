import pandas as pd
import os

# Definir las carpetas base
base_input_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/all"
base_output_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/metrics"

# Función para obtener los archivos CSV en un directorio
def get_csv_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.csv')]

# Función para evaluar un archivo CSV
def evaluate_csv(input_file, output_file):
    # Leer el archivo CSV
    df = pd.read_csv(input_file)
    
    # Excluir la última fila (global accuracy)
    df = df.iloc[:-1]
    
    # Contar los ejemplos correctos
    correctos = df[df['accuracy'] == 1].shape[0]
    
    # Contar los errores
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

# Función principal
def main():
    # Pedir al usuario que elija la fase (train, val, testdev)
    phase = input("Selecciona la fase (train, val, testdev): ").strip().lower()
    if phase not in ["train", "val", "testdev"]:
        print("Fase no válida. Debe ser 'train', 'val' o 'testdev'.")
        return

    # Definir las rutas de entrada y salida según la fase seleccionada
    input_folder = os.path.join(base_input_folder, phase)
    output_folder = os.path.join(base_output_folder, phase)

    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Obtener lista de archivos CSV
    csv_files = get_csv_files(input_folder)
    if not csv_files:
        print(f"No hay archivos CSV en {input_folder}")
        return

    # Preguntar si se quieren procesar todos los archivos
    process_all = input("¿Quieres evaluar todos los archivos CSV? (s/n): ").strip().lower()
    
    if process_all == 's':
        for file_name in csv_files:
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)
            evaluate_csv(input_file, output_file)
    else:
        # Si el usuario no elige procesar todos, permite seleccionar uno
        print("Lista de archivos disponibles:")
        for idx, file in enumerate(csv_files):
            print(f"{idx}: {file}")

        idx = int(input("Selecciona el índice del archivo CSV: "))
        file_name = csv_files[idx]
        
        input_file = os.path.join(input_folder, file_name)
        output_file = os.path.join(output_folder, file_name)
        evaluate_csv(input_file, output_file)

if __name__ == "__main__":
    main()
