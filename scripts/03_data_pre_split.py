from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# Load parameters
params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
SYMBOLS = params['DATA_ACQUISITON']['SYMBOLS']
PATH_FIGURE = params['DATA_UNDERSTANDING']['FIGURE_PATH']

# Load raw data
df_raw = pd.read_parquet(f'{PATH_BARS}/stock_data.parquet')

# Feature Engineering
ema_windows = [5, 10, 15, 30]

all_features = []

# Loop for each symbol to calculate EMAs, slopes, accelerations, and normalize features
for symbol in SYMBOLS:
    print(f"Processing {symbol}...")
    df = df_raw[df_raw["symbol"] == symbol].copy()

    # Calculate EMAs
    for w in ema_windows:
        df[f'ema_{w}'] = df['close'].ewm(span=w, adjust=False).mean()
        df[f'ema_{w}_slope'] = df[f'ema_{w}'].diff()
        df[f'ema_{w}_accel'] = df[f'ema_{w}_slope'].diff()

    # Normalize features - commented out because normalization will be done after train-test split
    #feature_cols = [f"ema_{w}" for w in ema_windows] + \
    #               [f"ema_{w}_slope" for w in ema_windows] + \
    #               [f"ema_{w}_accel" for w in ema_windows]

    #scaler = StandardScaler()
    #df[[f"{col}_norm" for col in feature_cols]] = scaler.fit_transform(df[feature_cols])

    # Save features for this symbol
    all_features.append(df)

# Combine all symbols' features into a single DataFrame
df_all = pd.concat(all_features, ignore_index=True)

# Save features to Parquet file
print('Saving features to parquet file...')
df_all.to_parquet(f"{PATH_BARS}/stock_features.parquet", index=False)

# Plot all EMA30 Accelerations for each symbol
print("Plotting all symbols' EMA30 Acceleration...")
plt.figure(figsize=(12,6))
for symbol in SYMBOLS:
    df = df_all[df_all["symbol"] == symbol]
    plt.plot(df["timestamp"], df["ema_30_accel"], label=f"{symbol} EMA30 Accel")

plt.legend()
plt.title("EMA30 Acceleration for All Symbols")
plt.xlabel("Date")
plt.ylabel("Acceleration")
plt.tight_layout()
plt.savefig(f"{PATH_FIGURE}/03_stock_ema30_accel.png")
plt.close()