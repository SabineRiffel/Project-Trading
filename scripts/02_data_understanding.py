import matplotlib.pyplot as plt
import pandas as pd
import pytz
import yaml
import random


# Plot stock performance for multiple symbols
# Load data acquisition parameters from YAML configuration file
print('Loading parameters...')
params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
START_DATE = pd.to_datetime(params['DATA_ACQUISITON']['START_DATE'])
END_DATE = pd.to_datetime(params['DATA_ACQUISITON']['END_DATE'])
SYMBOLS = params['DATA_ACQUISITON']['SYMBOLS']
PATH_FIGURE = params['DATA_UNDERSTANDING']['FIGURE_PATH']

# Read the stock data from a CSV file
print('Loading data...')
df = pd.read_parquet(f'{PATH_BARS}/stock_data.parquet')

# Ensure the 'timestamp' column is in datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Plot the closing price over time
print('Plotting stock performance...')
plt.figure(figsize=(14, 6))

# Plot each symbol's closing price
for symbol in SYMBOLS:
    print(f'Processing {symbol}...')
    df_symbol = df[df['symbol'] == symbol]
    plt.plot(df_symbol['timestamp'], df_symbol['close'], label=f'{symbol} Close Price', linewidth=1)

# Customize the plot
plt.title(f"Stock Performance Over Time", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f'{PATH_FIGURE}/02_stock_performance_plot.png'); plt.close()

# Plot news count per symbol
# Read the news data from a CSV file
print('Loading news data...')
#df_news = pd.read_csv(f'{PATH_BARS}/news_data.csv')
df_news = pd.read_parquet(f'{PATH_BARS}/news_parquet.csv')

# Plot the count of news articles per symbol
print('Plotting news count per symbol...')
news_counts = df_news['symbol'].value_counts()
plt.figure(figsize=(10, 6))
news_counts.plot(kind='bar', color='skyblue')
plt.title("Count of News per Symbol", fontsize=14)
plt.xlabel("Symbol", fontsize=12)
plt.ylabel("News", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"{PATH_FIGURE}/02_news_count_per_symbol_plot.png"); plt.close()

