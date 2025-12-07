import yaml
from datetime import datetime
import pandas as pd

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
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest
from bs4 import BeautifulSoup

print('Downloading news data...')
news_client = NewsClient(api_key=API_KEY, secret_key=SECRET_KEY)

request_params = NewsRequest(
    symbols=SYMBOLS[0],
    start=START_DATE,
    end=END_DATE,
    include_content=True
)

# Fetch news data
news = news_client.get_news(request_params)

articles = news.dict()

news_list =articles['news']
news_data = []
print(news_list)

# Process each article to extract plain text from HTML content
for article in news_list:
    # Convert HTML content to plain text
    raw_html = article.get("content", "")
    plain_text = BeautifulSoup(raw_html, "html.parser").get_text(strip=True)

    news_data.append({
        "id": article.get("id"),
        "headline": article.get("headline"),
        "summary": article.get("summary"),
        "url": article.get("url"),
        "created_at": article.get("created_at"),
        "updated_at": article.get("updated_at"),
        #"content": article.get("content"),
        "content": plain_text,
        "source": article.get("source")#,
        #"symbols": ",".join(article.get("symbols", []))
    })

#
df = pd.DataFrame(news_data)

# ---------------------------------------------------------
# Save DataFrame as CSV
# ---------------------------------------------------------

df.to_csv(f'{PATH_BARS}/{SYMBOLS[0]}.csv', index=False, encoding="utf-8")
print("Done.")


