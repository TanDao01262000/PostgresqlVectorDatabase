import pandas as pd

df = pd.read_csv('30-movies.csv')
new_columns = df.columns[-1]
df_filtered = df[new_columns]
df_filtered.to_csv('30-movies-only-text.csv')