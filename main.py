import pandas as pd

table = pd.read_csv('./csv/crime.csv')

columns = table[['Year', 'Month', 'Incident']]

columns.to_csv('crime.csv', index=False)

print(table)