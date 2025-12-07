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

all_data = []

# Loop over all symbols
for symbol in SYMBOLS:
    print(f'Processing {symbol}...')
    request_params = NewsRequest(
        symbols=symbol,
        start=START_DATE,
        end=END_DATE,
        include_content=True
    )

    # Fetch news data for this symbol
    news = news_client.get_news(request_params)
    articles = news.dict()
    news_list = articles.get('news', [])

    # Process each article
    for article in news_list:
        raw_html = article.get("content", "")
        plain_text = BeautifulSoup(raw_html, "html.parser").get_text(strip=True)

        all_data.append({
            "symbol": symbol,
            "timestamp": article.get("created_at"),
            "headline": article.get("headline"),
            "content": plain_text,
            "summary": article.get("summary"),
            "url": article.get("url")
        })

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Convert 'created_at' to Eastern Time
eastern = pytz.timezone("US/Eastern")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(eastern)

df = df[(df['timestamp'] >= START_DATE.astimezone(eastern)) &
        (df['timestamp'] <= END_DATE.astimezone(eastern))]

#df.to_csv(f'{PATH_BARS}/news_data.csv', index=False, encoding="utf-8")
df.to_parquet(f'{PATH_BARS}/news_data.parquet', index=False)
print("News data acquisition complete.")