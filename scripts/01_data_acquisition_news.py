from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import pytz
import yaml

# Download stock bar data from Alpaca API and save as Parquet file
# Load API credential from YAML configuration file
print('Loading API keys...')
keys = yaml.safe_load(open("../data/conf/keys.yaml"))
API_KEY = keys['KEYS']['APCA-API-KEY-ID-Data']
SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY-Data']

# Load data acquisition parameters from YAML configuration file
print('Loading parameters...')
params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
START_DATE = datetime.strptime(params['DATA_ACQUISITON']['START_DATE'], "%Y-%m-%d")
END_DATE = datetime.strptime(params['DATA_ACQUISITON']['END_DATE'], "%Y-%m-%d")
SYMBOLS = params['DATA_ACQUISITON']['SYMBOLS']

# Download news data from Alpaca API and save as CSV file
# Initialize the NewsClient with API credentials
print('Downloading news data...')
news_client = NewsClient(api_key=API_KEY, secret_key=SECRET_KEY)

# Create a request object for news data
request_params = NewsRequest(
    symbols=SYMBOLS[0],
    start=START_DATE,
    end=END_DATE,
    include_content=True
)

# Fetch news data
news = news_client.get_news(request_params)

# Convert news data to a list of dictionaries
articles = news.dict()

# Extract the list of news articles
news_list =articles['news']
news_data = []

# Process each article to extract plain text from HTML content
for article in news_list:
    # Convert HTML content to plain text
    raw_html = article.get("content", "")
    plain_text = BeautifulSoup(raw_html, "html.parser").get_text(strip=True)

    news_data.append({
        "timestamp": article.get("created_at"),
        "headline": article.get("headline"),
        "content": plain_text,
        "summary": article.get("summary"),
        "url": article.get("url")
    })

# Save the processed news data to a CSV file
df = pd.DataFrame(news_data)

# Convert 'created_at' to Eastern Time
eastern = pytz.timezone("US/Eastern")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(eastern)

df = df[(df['timestamp'] >= START_DATE.astimezone(eastern)) &
        (df['timestamp'] <= END_DATE.astimezone(eastern))]

df.to_csv(f'{PATH_BARS}/{SYMBOLS[0]}_news.csv', index=False, encoding="utf-8")
print("News data acquisition complete.")