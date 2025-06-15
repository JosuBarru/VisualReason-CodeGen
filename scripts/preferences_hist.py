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
    input_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/all/testdev"
    output_folder = "/sorgin1/users/jbarrutia006/viper/results/gqa/metrics/testdev"
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

        # Omitir la última fila (se asume que es un resumen TOTAL que no quieres procesar)
        if len(df) > 0:
            df = df.iloc[:-1]
            
        print(f"{file} tiene {len(df)} instancias")  # Print the number of rows in the CSV
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

    # Definir el orden fijo de las combinaciones de ticks (según la estructura de la tabla)
    ordered_combinations = [
        (0, 0, 0, 1),  # Solo Correctos
        (1, 0, 0, 1),  # Error Codigo y Correctos
        (0, 1, 0, 1),  # Error Ejecución y Correctos
        (1, 1, 0, 1),  # Error Codigo, Error Ejecución y Correctos
        (0, 0, 1, 1),  # Error Sem/Inf y Correctos
        (1, 0, 1, 1),  # Error Codigo, Error Sem/Inf y Correctos
        (0, 1, 1, 1),  # Error Ejecución, Error Sem/Inf y Correctos
        (1, 1, 1, 1),   # Todos los errores/correctos marcados

        (1, 0, 0, 0),  # Solo Error Codigo
        (0, 1, 0, 0),  # Solo Error Ejecución
        (1, 1, 0, 0),  # Error Codigo y Error Ejecución
        (0, 0, 1, 0),  # Solo Error Sem/Inf
        (1, 0, 1, 0),  # Error Codigo y Error Sem/Inf
        (0, 1, 1, 0),  # Error Ejecución y Error Sem/Inf
        (1, 1, 1, 0)  # Error Codigo, Error Ejecución y Error Sem/Inf
    ]
    
    # Construir la tabla de resultados siguiendo el orden de ticks definido
    data = []
    for comb in ordered_combinations:
        count = combination_counts.get(comb, 0)
        row = {
            "Error Codigo": "✓" if comb[0] == 1 else "",
            "Error Ejecución": "✓" if comb[1] == 1 else "",
            "Error Sem/Inf": "✓" if comb[2] == 1 else "",
            "Correctos": "✓" if comb[3] == 1 else "",
            "NUMERO": count
        }
        data.append(row)
    
    df_results = pd.DataFrame(data)
    
    total_instances = sum(row["NUMERO"] for row in data)
    total_row = {
        "Error Codigo": "TOTAL",
        "Error Ejecución": "",
        "Error Sem/Inf": "",
        "Correctos": "",
        "NUMERO": total_instances
    }
    # Agregar la fila de totales usando pd.concat (df.append está deprecado)
    df_results = pd.concat([df_results, pd.DataFrame([total_row])], ignore_index=True)
    
    output_file = os.path.join(output_folder, "combined_metrics.csv")
    df_results.to_csv(output_file, index=False)
    print(f"Resultados guardados en: {output_file}")

    html_output_file = os.path.join(output_folder, "combined_metrics.html")
    df_results.to_html(html_output_file, index=False)
    print(f"\nTabla HTML guardada en: {html_output_file}")

if __name__ == "__main__":
    main()