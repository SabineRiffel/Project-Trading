import matplotlib.pyplot as plt
import yaml
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

# Load parameters
params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
SYMBOLS = params['DATA_ACQUISITON']['SYMBOLS']
PATH_FIGURE = params['DATA_UNDERSTANDING']['FIGURE_PATH']

# Load engineered features
df_stock = pd.read_parquet(f"{PATH_BARS}/stock_features.parquet")
df_news = pd.read_parquet(f"{PATH_BARS}/news_features.parquet")

df_stock["timestamp"] = pd.to_datetime(df_stock["timestamp"], utc=True)
df_news["timestamp"] = pd.to_datetime(df_news["timestamp"], utc=True)

df_merged = pd.merge_asof(
    df_news.sort_values("timestamp"),
    df_stock.sort_values("timestamp"),
    on="timestamp",
    by="symbol",
    direction="backward", # Uses the last available bar before the news timestamp
    tolerance=pd.Timedelta("5min")  # Allow a tolerance of 5 minutes
)

splits = {
  "train": ["2022-01-01 00:00:00", "2024-12-31 23:59:59"],
  "val":   ["2025-01-01 00:00:00", "2025-03-31 23:59:59"],
  "test":  ["2025-04-01 00:00:00", "2025-06-30 23:59:59"]
}

with open(f"{PATH_BARS}/splits.json", "w") as f:
    json.dump(splits, f, indent=2)

# Define feature columns
ema_windows = [5, 10, 15, 30]
feature_cols = [f"ema_{w}" for w in ema_windows] + \
               [f"ema_{w}_slope" for w in ema_windows] + \
               [f"ema_{w}_accel" for w in ema_windows] + \
               ["close", "volume", "vwap", "sentiment"]

target_col = "future_return_30m"

# Target: future_return_30m

df_merged = df_merged.set_index("timestamp").sort_index()

# Split data into train, val, test sets
train = df_merged.loc[splits["train"][0]:splits["train"][1]]
val   = df_merged.loc[splits["val"][0]:splits["val"][1]]
test  = df_merged.loc[splits["test"][0]:splits["test"][1]]

# Save features and target variables
X_train, y_train = train[feature_cols], train[target_col]
X_val, y_val = val[feature_cols], val[target_col]
X_test, y_test = test[feature_cols], test[target_col]

# Normalize features using StandardScaler - commented out for tree-based models
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(train[feature_cols])
#X_val_scaled  = scaler.transform(val[feature_cols])
#X_test_scaled  = scaler.transform(test[feature_cols])

# Save the scaled features back to DataFrames
X_train = pd.DataFrame(X_train, columns=feature_cols, index=X_train.index)
X_val = pd.DataFrame(X_val, columns=feature_cols, index=X_val.index)
X_test = pd.DataFrame(X_test, columns=feature_cols, index=X_test.index)

# Save to parquet files
X_train.to_parquet(f"{PATH_BARS}/news model/X_train_news.parquet", index=True)
y_train.to_frame().to_parquet(f"{PATH_BARS}/news model/y_train_news.parquet", index=True)
X_val.to_parquet(f"{PATH_BARS}/news model/X_val_news.parquet", index=True)
y_val.to_frame().to_parquet(f"{PATH_BARS}/news model/y_val_news.parquet", index=True)
X_test.to_parquet(f"{PATH_BARS}/news model/X_test_news.parquet", index=True)
y_test.to_frame().to_parquet(f"{PATH_BARS}/news model/y_test_news.parquet", index=True)

# Plot Sentiment vs. Future Return for each symbol (Train set, sentiment ≠ 0)
symbols = SYMBOLS
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, symbol in enumerate(symbols):
    ax = axes[i]
    # Filter only data for the symbol AND sentiment ≠ 0
    df_symbol = train[(train["symbol"] == symbol) & (train["sentiment"] != 0)]

    ax.scatter(df_symbol["sentiment"], df_symbol["future_return_30m"],
               alpha=0.3, c="steelblue")
    ax.set_title(symbol)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Future Return 30m")
    ax.grid(True, linestyle="--", alpha=0.6)

# Adjust layout and show plot
print("Saving plot for price reaction to news events...")
plt.suptitle("Sentiment vs. Future Return (Train) – w/o Neutral Sentimente", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
