import pandas as pd

table = pd.read_csv('../csv/news.csv', encoding='ISO-8859-1')

columns = table['headline', '']

columns.to_csv('headlines.csv', index=False)

print(table)