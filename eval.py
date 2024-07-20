import pandas as pd

df = pd.read_csv('/sorgin1/users/jbarrutia006/viper/results/gqa/all/testdev/results_all_gemma.csv')

# Contar los ejemplos correctos
correctos = df[df['Answer'] == df['truth_answers']].shape[0]

# Contar las instancias que contienen "Error" en la columna 'Answer'
errores = df[df['Answer'].str.contains('Error', na=False)].shape[0]

# Contar el resto de las instancias
resto = df.shape[0] - correctos - errores

print(f'Cantidad de ejemplos correctos: {correctos}')
print(f'Cantidad de instancias con errores: {errores}')
print(f'Cantidad del resto de las instancias: {resto}')
