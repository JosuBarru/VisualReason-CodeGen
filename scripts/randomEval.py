import pandas as pd

df = pd.read_csv('/sorgin1/users/jbarrutia006/viper/results/gqa/all/testdev/results_all_70b.csv')

#Me quedo con las instancias que no dan error de ejecucion pero que la respuesta no es correcta
resto_df = df[(df['accuracy'] == 0) & ~df['Answer'].str.contains('Error', na=False)]

resto_aleatorio = resto_df.sample(n=20, random_state=42)

resto_aleatorio.to_csv('/sorgin1/users/jbarrutia006/viper/results/gqa/all/testdev/results_all_analysis_sample_70b.csv', header=True, index=False, encoding='utf-8')
