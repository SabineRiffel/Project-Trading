import matplotlib.pyplot as plt
import pandas as pd
import pytz
import yaml
import random


# Plot stock performance for AAPL
# Load data acquisition parameters from YAML configuration file
print('Loading parameters...')
params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
START_DATE = pd.to_datetime(params['DATA_ACQUISITON']['START_DATE'])
END_DATE = pd.to_datetime(params['DATA_ACQUISITON']['END_DATE'])
SYMBOLS = params['DATA_ACQUISITON']['SYMBOLS']
PATH_FIGURE = params['DATA_UNDERSTANDING']['FIGURE_PATH']

# Read the stock price data from a Parquet file
print('Loading data...')
df = pd.read_parquet(f'{PATH_BARS}/{SYMBOLS[0]}.parquet')

# Ensure the 'timestamp' column is in datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Plot the closing price over time
print('Plotting stock performance...')
plt.figure(figsize=(14, 6))
plt.plot(df['timestamp'], df['close'], label='AAPL Close Price', color='blue', linewidth=1)
plt.title(f"AAPL Stock Performance Over Time", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f'{PATH_FIGURE}/02_stock_performance_plot.png'); plt.close()


# Plot news count for AAPL
# Read the news data from a CSV file
print('Loading news data...')
df_news = pd.read_csv(f'{PATH_BARS}/{SYMBOLS[0]}_news.csv')
df_news['timestamp'] = pd.to_datetime(df_news['timestamp'], utc=True).dt.tz_convert("US/Eastern")

# Filter news articles within the specified date range
eastern = pytz.timezone("US/Eastern")
START_DATE = eastern.localize(START_DATE)
END_DATE = eastern.localize(END_DATE)
mask = (df_news['timestamp'] >= START_DATE) & (df_news['timestamp'] <= END_DATE)
df_news = df_news.loc[mask]

# Count the number of news articles per day
news_per_day = df_news.groupby(df_news['timestamp'].dt.date).size()

# Plot the number of news articles over time
print('Plotting news count...')
plt.figure(figsize=(14, 6))
plt.plot(news_per_day.index, news_per_day.values, label='Apple News Count', color='purple', linewidth=1)
plt.title("Apple News Count Over Time", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Number of Articles", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f"{PATH_FIGURE}/02_news_count_plot.png"); plt.close()


# Combined plot: Stock price vs News count
# Prepare data for combined plot
df['date'] = df['timestamp'].dt.date
daily_close = df.groupby('date')['close'].last().reset_index()
news_per_day = df_news.groupby(df_news['timestamp'].dt.date).size().reset_index(name='news_count')
news_per_day.rename(columns={'timestamp': 'date'}, inplace=True)
merged = pd.merge(daily_close, news_per_day, on='date', how='inner')

# Plot combined stock price and news count
print('Plotting combined stock price and news count...')
plt.figure(figsize=(14, 6))
ax1 = plt.gca()
ax1.plot(merged['date'], merged['close'], color='blue', label='AAPL Close Price')
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Price (USD)", fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.plot(merged['date'], merged['news_count'], color='purple', label='Apple News Count')
ax2.set_ylabel("Number of Articles", fontsize=12, color='purple')
ax2.tick_params(axis='y', labelcolor='purple')
plt.title("AAPL Stock Price vs News Count Over Time", fontsize=14)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"{PATH_FIGURE}/02_stock_vs_news_plot.png"); plt.close()


# Plot VWAP vs Close Price
print('Calculating and plotting VWAP vs Close Price...')
# VWAP Calculation (intraday)
def calc_vwap(x):
    cum_pv = (x["close"] * x["volume"]).cumsum()
    cum_vol = x["volume"].cumsum()
    return cum_pv / cum_vol

# Apply VWAP calculation for each day
df["daily_vwap"] = df.groupby("date", group_keys=False).apply(calc_vwap)

# Plot intraday VWAP vs Close Price for a sample day
sample_day = random.choice(df["date"].unique())
day_data = df[df["date"] == sample_day]

# Plot intraday VWAP vs Close Price
plt.figure(figsize=(12,6))
plt.plot(day_data["timestamp"], day_data["close"], label="Close Price", color="blue")
plt.plot(day_data["timestamp"], day_data["daily_vwap"], label="VWAP (intraday)", color="purple", linestyle="--")
plt.title(f"Intraday VWAP vs Close Price für {SYMBOLS[0]} am {sample_day}")
plt.xlabel("Hour")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.savefig(f'{PATH_FIGURE}/02_vwap_vs_close.png'); plt.close()

