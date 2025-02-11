import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

def load_metrics(file_path):
    df = pd.read_csv(file_path)
    return df.iloc[0, 1], df.iloc[1, 1], df.iloc[2, 1]  #Errores Sintácticos, Errores Semánticos o Inf, Correct

def plot_confusion_matrix(model1_data, model2_data, model1_name, model2_name):
    labels = ['Errores Sintácticos', 'Errores Semánticos o Inferencia', 'Correcto']
    matrix = pd.DataFrame([model1_data, model2_data], index=[model1_name, model2_name], columns=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
    plt.xlabel("Categoría de Error")
    plt.ylabel("Modelo")
    plt.title("Matriz de Confusión Comparativa")
    plt.show()

# Obtener archivos disponibles
metrics_dir = "/sorgin1/users/jbarrutia006/viper/results/gqa/metrics/train"
csv_files = [f for f in os.listdir(metrics_dir) if f.endswith('.csv')]

if len(csv_files) < 2:
    print("No hay suficientes archivos CSV para comparar.")
else:
    print("Archivos disponibles:")
    for i, file in enumerate(csv_files):
        print(f"{i}: {file}")
    
    # Seleccionar los dos archivos
    idx1 = int(input("Ingrese el índice del primer archivo: "))
    idx2 = int(input("Ingrese el índice del segundo archivo: "))
    
    file1, file2 = csv_files[idx1], csv_files[idx2]
    model1_name, model2_name = file1.replace(".csv", ""), file2.replace(".csv", "")
    
    # Cargar métricas
    model1_data = load_metrics(os.path.join(metrics_dir, file1))
    model2_data = load_metrics(os.path.join(metrics_dir, file2))
    
    # Generar y mostrar matriz de confusión
    plot_confusion_matrix(model1_data, model2_data, model1_name, model2_name)
