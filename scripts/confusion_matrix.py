import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Cosas que cambiar
metrics_dir = "/sorgin1/users/jbarrutia006/viper/results/gqa/all/train"
save_dir = "/sorgin1/users/jbarrutia006/viper/results/gqa/metrics/train"
#################


def get_csv_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.csv')]

def select_csv_files(csv_files):
    print("Lista de archivos disponibles:")
    for idx, file in enumerate(csv_files):
        print(f"{idx}: {file}")
    
    idx1 = int(input("Selecciona el índice del primer archivo CSV: "))
    idx2 = int(input("Selecciona el índice del segundo archivo CSV: "))
    
    return csv_files[idx1], csv_files[idx2]

def categorize_instance(answer, accuracy):
    if accuracy == 1:
        return 3  # Correcto
    elif isinstance(answer, str):
        if answer.startswith('Error Codigo'):
            return 0  # Error Código
        elif answer.startswith('Error Ejecucion'):
            return 1  # Error Ejecución
    return 2  # Error Inferencia/Semántico

def compute_confusion_matrix(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    matriz = np.zeros((4, 4), dtype=int)
    
    for _, row1 in df1.iterrows():
        sample_id = row1['sample_id']
        row2 = df2[df2['sample_id'] == sample_id]
        
        if not row2.empty:
            row2 = row2.iloc[0]
            idx1 = categorize_instance(row1['Answer'], row1['accuracy'])
            idx2 = categorize_instance(row2['Answer'], row2['accuracy'])
            matriz[idx1][idx2] += 1
    
    return matriz

def plot_confusion_matrix(matrix, model1_name, model2_name):
    labels = ['Errores Código', 'Errores Ejecución', 'Errores Semánticos o Inferencia', 'Correcto']
    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
    
    conf_mat_dir = os.path.join(save_dir, "conf_mat")
    os.makedirs(conf_mat_dir, exist_ok=True)
    
    plot_filename = f"{model1_name}_vs_{model2_name}.png"
    plot_path = os.path.join(conf_mat_dir, plot_filename)
    
    plt.figure(figsize=(12, 10))  # Aumentar tamaño
    sns.heatmap(df_matrix, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
    
    plt.xlabel(f"Modelo: {model2_name}", fontsize=14)
    plt.ylabel(f"Modelo: {model1_name}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.title("Matriz de Confusión Comparativa", fontsize=16)
    plt.tight_layout()  # Ajusta automáticamente para evitar superposiciones
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Matriz de confusión guardada en: {plot_path}")

def main():
    csv_files = get_csv_files(metrics_dir)
    file1, file2 = select_csv_files(csv_files)
    file1_path = os.path.join(metrics_dir, file1)
    file2_path = os.path.join(metrics_dir, file2)
    
    matriz_confusion = compute_confusion_matrix(file1_path, file2_path)
    
    plot_confusion_matrix(matriz_confusion, file1, file2)


if __name__ == "__main__":
    main()
