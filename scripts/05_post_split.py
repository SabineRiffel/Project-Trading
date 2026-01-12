import matplotlib.pyplot as plt
import yaml
import pandas as pd
import json

# Load parameters
params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
SYMBOLS = params['DATA_ACQUISITON']['SYMBOLS']
PATH_FIGURE = params['DATA_UNDERSTANDING']['FIGURE_PATH']
SPLITS = params['DATA_SPLITS']

# Load engineered features
df_stock = pd.read_parquet(f"{PATH_BARS}/stock_features.parquet")
df_news = pd.read_parquet(f"{PATH_BARS}/news_features.parquet")

df_stock["timestamp"] = pd.to_datetime(df_stock["timestamp"], utc=True).dt.floor("min")
df_news["timestamp"] = pd.to_datetime(df_news["timestamp"], utc=True).dt.floor("min")

df_merged = pd.merge(
    df_news.sort_values("timestamp"),
    df_stock.sort_values("timestamp"),
    on=["timestamp", "symbol"],
    how="inner"
)

# Save splits to JSON for reference
with open(f"{PATH_BARS}/splits.json", "w") as f:
    json.dump(SPLITS, f, indent=2)

# One-hot encoding für Sentiment
df_merged = pd.get_dummies(df_merged, columns=["sentiment_label"], prefix="sentiment")

# Define feature columns
ema_windows = [5, 10, 15, 30]
feature_cols = [f"ema_{w}" for w in ema_windows] + \
               [f"ema_{w}_slope" for w in ema_windows] + \
               [f"ema_{w}_accel" for w in ema_windows] + \
               ["close", "volume", "vwap", "volume_spike"] + \
               [col for col in df_merged.columns if col.startswith("sentiment_")]

target_col_future_return = "future_return_30m"
target_col_future_direction = "target_direction"

df_merged = df_merged.set_index("timestamp").sort_index()

# Split data into train, val, test sets
train = df_merged.loc[SPLITS['TRAIN'][0]:SPLITS['TRAIN'][1]]
val   = df_merged.loc[SPLITS['VAL'][0]:SPLITS['VAL'][1]]
test  = df_merged.loc[SPLITS['TEST'][0]:SPLITS['TEST'][1]]

# Save features and target variables
X_train, y_train = train[feature_cols], train[[target_col_future_return,target_col_future_direction, "symbol"]]
X_val, y_val = val[feature_cols], val[[target_col_future_return, target_col_future_direction, "symbol"]]
X_test, y_test = test[feature_cols], test[[target_col_future_return, target_col_future_direction, "symbol"]]

# Save the scaled features back to DataFrames
X_train = pd.DataFrame(X_train, columns=feature_cols, index=X_train.index)
X_val = pd.DataFrame(X_val, columns=feature_cols, index=X_val.index)
X_test = pd.DataFrame(X_test, columns=feature_cols, index=X_test.index)

X_train.to_parquet(f"{PATH_BARS}/news model/X_train_news.parquet", index=True)
y_train.to_parquet(f"{PATH_BARS}/news model/y_train_news.parquet", index=True)
X_val.to_parquet(f"{PATH_BARS}/news model/X_val_news.parquet", index=True)
y_val.to_parquet(f"{PATH_BARS}/news model/y_val_news.parquet", index=True)
X_test.to_parquet(f"{PATH_BARS}/news model/X_test_news.parquet", index=True)
y_test.to_parquet(f"{PATH_BARS}/news model/y_test_news.parquet", index=True)

symbols = SYMBOLS
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, symbol in enumerate(symbols):
    ax = axes[i]
    df_symbol = train[train["symbol"] == symbol].copy()

    # Reconstruct sentiment category from one-hot encoded columns
    df_symbol["sentiment_category"] = (
        df_symbol[["sentiment_-1", "sentiment_0", "sentiment_1"]]
        .idxmax(axis=1)
        .str.replace("sentiment_", "")
    )

    # Boxplot for this symbol
    df_symbol.boxplot(column="future_return_30m", by="sentiment_category", ax=ax)
    ax.set_title(symbol)
    ax.set_xlabel("Sentiment category")
    ax.set_ylabel("Future Return 30m")
    ax.grid(True, linestyle="--", alpha=0.6)

# Adjust layout
plt.suptitle("Future Return 30m by Sentiment Category per Symbol", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()