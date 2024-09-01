import os
from dotenv import load_dotenv
from newsapi import NewsApiClient

# load env variables
load_dotenv()

# api key
api_key = os.getenv('API_KEY')

# test
# print(api_key)

# Initialize newsapi
newsapi = NewsApiClient(api_key='93621f25419542a297080d27d29085c9')

# retrieve last months news
# need to edit so it uses current date, not august date
past_month_articles = newsapi.get_everything(q='finance',
                                      from_param='2024-08-01',
                                      to='2024-08-31',
                                      language='en')

# all titles from last month
past_month_titles = []

# add all titles from last month to titles array
for article in past_month_articles['articles']:
    past_month_titles.append(article['title'])

# test
print(past_month_titles)


