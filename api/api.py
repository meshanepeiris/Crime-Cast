import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
from datetime import datetime
from dateutil.relativedelta import relativedelta

# load env variables
load_dotenv()

# api key
api_key = os.getenv('API_KEY')

# datetime
today = datetime.now().date()
one_month_ago = datetime.now() - relativedelta(months=1)

# test datetime
# print(one_month_ago)

# test api
# print(api_key)

# Initialize newsapi
newsapi = NewsApiClient(api_key=api_key)

past_month_articles = newsapi.get_everything(q='finance',
                                      from_param=one_month_ago,
                                      to=today,
                                      language='en')

past_month_titles = []
past_month_descriptions = []

for article in past_month_articles['articles']:
    past_month_titles.append(article['title'])
    past_month_descriptions.append(article['description'])






