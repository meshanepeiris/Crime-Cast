import pandas as pd

df = pd.read_csv('../csv/news.csv', encoding='ISO-8859-1')

df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Step 3: Sort the DataFrame by the date column
df_sorted = df.sort_values(by='date')

# Step 4 (Optional): Save the sorted DataFrame back to a CSV file
df_sorted.to_csv('sorted_date_news.csv', index=False)