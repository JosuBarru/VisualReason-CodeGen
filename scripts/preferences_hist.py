import os
import pandas as pd

def get_csv_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.csv')]

def select_multiple_csv_files(csv_files):
    selected_files = []
    print("Lista de archivos disponibles:")
    for idx, file in enumerate(csv_files):
        print(f"{idx}: {file}")
    while True:
        user_input = input("Selecciona el índice del archivo CSV (o escribe 'fin' para terminar): ")
        if user_input.lower() == 'fin':
            break
        try:
            idx = int(user_input)
            if idx < 0 or idx >= len(csv_files):
                print("Índice inválido, intenta de nuevo.")
            else:
                selected_files.append(csv_files[idx])
        except ValueError:
            print("Entrada no válida. Por favor, introduce un número o 'fin'.")
    return selected_files

def get_classification_bits(answer, accuracy):
    """
    Retorna una lista de 4 bits correspondiente a las siguientes categorías:
      [Error Codigo, Error Ejecución, Error Sem/Inf, Correctos]
    Se asigna 1 en la posición correspondiente si la instancia cumple la condición.
    """
    if accuracy == 1:
        return [0, 0, 0, 1]
    elif isinstance(answer, str):
        if answer.startswith('Error Codigo'):
            return [1, 0, 0, 0]
        elif answer.startswith('Error Ejecucion'):
            return [0, 1, 0, 0]
    return [0, 0, 1, 0]

def main():
    input_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/all/"
    output_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/metrics/"
    os.makedirs(output_folder, exist_ok=True)
    
    csv_files = get_csv_files(input_folder)
    if not csv_files:
        print("No se encontraron archivos CSV en el directorio.")
        return
    
    selected_files = select_multiple_csv_files(csv_files)
    if not selected_files:
        print("No se seleccionó ningún archivo. Saliendo.")
        return
    
    # Diccionario para almacenar la clasificación combinada de cada instancia
    # key: sample_id, value: lista de 4 bits
    instance_classifications = {}
    
    for file in selected_files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            sample_id = row['sample_id']
            bits = get_classification_bits(row['Answer'], row['accuracy'])
            if sample_id not in instance_classifications:
                instance_classifications[sample_id] = bits
            else:
                # Operación OR: para cada categoría, se marca 1 si al menos un fichero la tiene
                instance_classifications[sample_id] = [
                    max(existing, new)
                    for existing, new in zip(instance_classifications[sample_id], bits)
                ]
    
    # Contar la frecuencia de cada combinación (representada como tupla de 4 bits)
    combination_counts = {}
    for bits in instance_classifications.values():
        bits_tuple = tuple(bits)
        combination_counts[bits_tuple] = combination_counts.get(bits_tuple, 0) + 1
        
    # Construir la tabla de resultados. Se mostrará un tick (✓) si la categoría está presente.
    data = []
    for bits_tuple, count in combination_counts.items():
        row = {
            "Error Codigo": "✓" if bits_tuple[0] == 1 else "",
            "Error Ejecución": "✓" if bits_tuple[1] == 1 else "",
            "Error Sem/Inf": "✓" if bits_tuple[2] == 1 else "",
            "Correctos": "✓" if bits_tuple[3] == 1 else "",
            "Numero": count
        }
        data.append(row)
    
    df_results = pd.DataFrame(data)
    # Ordenar la tabla (por ejemplo, de mayor a menor número de instancias)
    df_results = df_results.sort_values(by="Numero", ascending=False)
    
    total_instances = sum(combination_counts.values())
    total_row = {
        "Error Codigo": "TOTAL",
        "Error Ejecución": "",
        "Error Sem/Inf": "",
        "Correctos": "",
        "Numero": total_instances
    }
    # Agregar la fila de totales
    df_results = df_results.append(total_row, ignore_index=True)
    
    output_file = os.path.join(output_folder, "combined_metrics.csv")
    df_results.to_csv(output_file, index=False)
    print(f"Resultados guardados en: {output_file}")

if __name__ == "__main__":
    main()
