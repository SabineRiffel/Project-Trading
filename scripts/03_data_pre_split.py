from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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

# Load the downloaded Parquet file into a DataFrame and copy it so raw data is preserved
print('Loading data from Parquet file...')
df_raw = pd.read_parquet(f'{PATH_BARS}/{SYMBOLS[0]}.parquet')
df = df_raw.copy()

# Normalize 'vwap' and 'volume' columns using StandardScaler
print('Normalizing vwap and volume columns...')
scaler = StandardScaler()
df[["vwap_norm", "volume_norm"]] = scaler.fit_transform(df[["vwap", "volume"]])

# Calculate Exponential Moving Averages (EMAs) for specified time windows
print('Calculating exponential moving averages (EMAs)...')
ema_windows = [5, 10, 15, 30, 60]

for w in ema_windows:
    df[f'ema_{w}'] = df['close'].ewm(span=w, adjust=False).mean()

# Calculate slopes and accelerations of the EMAs
for w in ema_windows:
    df[f"ema_{w}_slope"] = df[f"ema_{w}"].diff()

for w in ema_windows:
    df[f"ema_{w}_accel"] = df[f"ema_{w}_slope"].diff()

feature_cols = [f"ema_{w}" for w in ema_windows] + \
               [f"ema_{w}_slope" for w in ema_windows] + \
               [f"ema_{w}_accel" for w in ema_windows]

scaler = StandardScaler()
df[[f"{col}_norm" for col in feature_cols]] = scaler.fit_transform(df[feature_cols])

# Save the DataFrame with features to a new Parquet file
print('Saving features to Parquet file...')
df.to_parquet(f'{PATH_BARS}/{SYMBOLS[0]}_features.parquet', index=False)
print('Feature engineering complete.')



# Plot all EMA Slopes
print("Plotting all EMA slopes...")
plt.figure(figsize=(12,6))
for w in ema_windows:
    plt.plot(df["timestamp"], df[f"ema_{w}_accel"], label=f"Accel EMA {w}")
plt.legend()
plt.title("EMA Accelerations (Trend Acceleration)")
plt.xlabel("Date")
plt.ylabel("Acceleration")
plt.tight_layout()
plt.savefig(f"{PATH_FIGURE}/03_ema_trend.png"); plt.close()

# Plot normalized VWAP and Volume
plt.figure(figsize=(12,6))
plt.plot(df["timestamp"], df["volume_norm"], label="Normalized Volume", color="orange")
plt.plot(df["timestamp"], df["vwap_norm"], label="Normalized VWAP", color="blue")
plt.title("Normalized VWAP and Volume over Time")
plt.xlabel("Time")
plt.ylabel("Normalized Values")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{PATH_FIGURE}/03_vwap_vs_volume.png"); plt.close()