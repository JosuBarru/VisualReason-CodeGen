import pandas as pd

df = pd.read_csv('/sorgin1/users/jbarrutia006/viper/results/okvqa/all/test/results_all_7b_mistral.csv')

# Contar los ejemplos correctos
correctos = df[df['accuracy'] == 1].shape[0]

# Contar las instancias que contienen "Error" en la columna 'Answer'
errores = df[df['Answer'].str.startswith('Error', na=False)].shape[0]

# Contar el resto de las instancias
resto = df.shape[0] - correctos - errores

print(f'Cantidad de instancias correctas: {correctos}')
print(f'Cantidad de instancias con errores sintacticos/ejecucion: {errores}')
print(f'Cantidad de instancias con errores semánticos o de inferencia: {resto}')
