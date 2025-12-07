from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import math

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
df_raw = pd.read_parquet(f'{PATH_BARS}/news_data.parquet')
df = df_raw.copy()

# VADER Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
df["sentiment"] = df["headline"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
df["sentiment_label"] = df["sentiment"].apply(
    lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
)

# TF-IDF Vectorization for Headlines
vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer.fit_transform(df["headline"])

# Prepare timestamp and date columns
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["date"] = pd.to_datetime(df["timestamp"].dt.date)

# Load price data to calculate daily returns
apple_prices = pd.read_parquet(f'{PATH_BARS}/stock_data.parquet')
apple_prices["timestamp"] = pd.to_datetime(apple_prices["timestamp"], utc=True)
apple_prices["date"] = pd.to_datetime(apple_prices["timestamp"].dt.date)

# Calculate daily returns
daily = apple_prices.groupby("date").agg({"open": "first", "close": "last"}).reset_index()
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

# Save the DataFrame with features to a new CSV file
print('Saving features to CSV file...')
df.to_csv(f'{PATH_BARS}/news_features.csv', index=False)
print('News feature engineering complete.')

# Plot the distribution of sentiment scores
plt.figure(figsize=(10, 6))

for symbol in SYMBOLS:
    sns.histplot(
        df[df['symbol'] == symbol]["sentiment"],
        bins=30, kde=True, stat="density", element="step",
        label=symbol, alpha=0.4
    )

plt.title("Sentiment Distribution Across Symbols")
plt.xlabel("Sentiment Score")
plt.ylabel("Density")
plt.legend(title="Symbol")
plt.tight_layout()
plt.savefig(f'{PATH_FIGURE}/03_news_sentiment_distribution_.png');
plt.close()

# Plot price reaction to news events
window = 30  # Minutes before and after the news event

# Load price data
prices = pd.read_parquet(f'{PATH_BARS}/stock_data.parquet')
prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)

# Count number of symbols to determine subplot grid size
n_symbols = len(SYMBOLS)
n_cols = 3
n_rows = math.ceil(n_symbols / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharey=True)

# Flatten axes array for easy indexing
axes = axes.flatten()

# Calculate and plot price reaction for each symbol
print("Calculating price reaction...")
for i, symbol in enumerate(SYMBOLS):
    df_symbol = df[df['symbol'] == symbol].copy()
    prices_symbol = prices[prices['symbol'] == symbol]

    event_returns = []
    # Iterate over each news event for the symbol
    for _, row in df_symbol.iterrows():
        ts = row['timestamp']
        sub = prices_symbol[(prices_symbol['timestamp'] >= ts - pd.Timedelta(minutes=window)) &
                            (prices_symbol['timestamp'] <= ts + pd.Timedelta(minutes=window))].copy()

        # Calculate normalized returns relative to the news event
        if len(sub) > 0:
            sub['minutes'] = (sub['timestamp'] - ts).dt.total_seconds() / 60
            sub['norm_return'] = (sub['close'] / sub.iloc[0]['close']) - 1
            event_returns.append(sub[['minutes', 'norm_return']])

    # Aggregate and plot the average normalized return curve
    if event_returns:
        avg_curve = pd.concat(event_returns).groupby('minutes')['norm_return'].mean()
        sns.lineplot(x=avg_curve.index, y=avg_curve.values, ax=axes[i])
        axes[i].axvline(0, color="purple", linestyle="--")
        axes[i].set_title(f"{symbol} News Price Reaction")
        axes[i].set_xlabel("Minutes relative to News")
        axes[i].set_ylabel("Price Reaction (Normalized Return)")

print("Saving plot for price reaction to news events...")
plt.tight_layout()
plt.savefig(f"{PATH_FIGURE}/03_news_price_reaction.png");plt.close()
