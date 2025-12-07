import pandas as pd
import matplotlib.pyplot as plt
import yaml

# Load parameters
params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
PATH_FIGURE = params['DATA_UNDERSTANDING']['FIGURE_PATH']

# Load raw data
df_raw = pd.read_parquet(f'{PATH_BARS}/stock_features.parquet')
df = df_raw.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
df = df.sort_index()

# Define data splits
splits = {
    "train": ["2022-01-01 00:00:00", "2024-12-31 23:59:59"],
    "val":   ["2025-01-01 00:00:00", "2025-03-31 23:59:59"],
    "test":  ["2025-04-01 00:00:00", "2025-06-30 23:59:59"]
}

# Save split definitions to JSON file
df.to_json(f"{PATH_BARS}/splits.json", orient="records", indent=2)

# Visualize the splits
train = df.loc[splits["train"][0]:splits["train"][1]]
val   = df.loc[splits["val"][0]:splits["val"][1]]
test  = df.loc[splits["test"][0]:splits["test"][1]]

# Plot close prices for each split to visually confirm the chronological separation
plt.figure(figsize=(8,3), dpi=100)
plt.plot(train.index, train["close"], label="Train",color="lightskyblue")
plt.plot(val.index, val["close"], label="Validation", color="steelblue")
plt.plot(test.index, test["close"], label="Test", color="midnightblue")
plt.title("Train/Val/Test split")
plt.tight_layout()
plt.savefig(f"{PATH_FIGURE}/04_stock_data_split.png");plt.close()





