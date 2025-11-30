from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml

# Load data acquisition parameters from YAML configuration file
print('Loading parameters...')
params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
START_DATE = datetime.strptime(params['DATA_ACQUISITON']['START_DATE'], "%Y-%m-%d")
END_DATE = datetime.strptime(params['DATA_ACQUISITON']['END_DATE'], "%Y-%m-%d")
SYMBOLS = params['DATA_ACQUISITON']['SYMBOLS']
PATH_FIGURE = params['DATA_UNDERSTANDING']['FIGURE_PATH']

# Load the downloaded CSV file into a DataFrame and copy it so raw data is preserved
print('Loading data from CSV file...')
df_raw = pd.read_csv(f'{PATH_BARS}/{SYMBOLS[0]}_news.csv')
df = df_raw.copy()

# VADER Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
df["sentiment"] = df["headline"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

# TF-IDF Vectorization for Headlines
vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer.fit_transform(df["headline"])

# Prepare timestamp and date columns
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["date"] = pd.to_datetime(df["timestamp"].dt.date)

# Load price data to calculate daily returns
apple_prices = pd.read_parquet(f'{PATH_BARS}/{SYMBOLS[0]}.parquet')
apple_prices["timestamp"] = pd.to_datetime(apple_prices["timestamp"], utc=True)
apple_prices["date"] = pd.to_datetime(apple_prices["timestamp"].dt.date)

# Calculate daily returns
daily = apple_prices.groupby("date").agg({"open":"first", "close":"last"}).reset_index()
daily["daily_return"] = (daily["close"] - daily["open"]) / daily["open"]

# Merge daily returns back to the news DataFrame
# Using merge_asof to return the next day's return for news articles if published after market close
df = pd.merge_asof(
    df.sort_values("date"),
    daily[["date", "daily_return"]].sort_values("date"),
    on="date",
    direction="forward"
)

# Label daily returns
df["label"] = df["daily_return"].apply(
    lambda x: "positive" if x > 0.01 else ("negative" if x < -0.01 else "neutral")
)
# Plot the distribution of sentiment scores
plt.figure(figsize=(8,6))
sns.histplot(df["sentiment"], bins=30, kde=True)
plt.title("Distribution of Sentiment Scores")
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.savefig(f'{PATH_FIGURE}/03_sentiment_distribution.png'); plt.close()



