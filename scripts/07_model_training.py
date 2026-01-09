import yaml
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Load parameters
print('Loading parameters...')
params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
SYMBOLS = params['DATA_ACQUISITON']['SYMBOLS']
PATH_FIGURE = params['DATA_UNDERSTANDING']['FIGURE_PATH']

# Load training, validation
print('Loading data sets...')
X_train = pd.read_parquet(f"{PATH_BARS}/news model/X_train_news.parquet")
y_train = pd.read_parquet(f"{PATH_BARS}/news model/y_train_news.parquet")
X_val = pd.read_parquet(f"{PATH_BARS}/news model/X_val_news.parquet")
y_val = pd.read_parquet(f"{PATH_BARS}/news model/y_val_news.parquet")

# Ensure consistent feature set
expected_features = [ "ema_5", "ema_10", "ema_15", "ema_30",
                      "ema_5_slope", "ema_10_slope", "ema_15_slope", "ema_30_slope",
                      "ema_5_accel", "ema_10_accel", "ema_15_accel", "ema_30_accel",
                      "close", "volume", "vwap", "volume_spike",
                      "sentiment_-1", "sentiment_0", "sentiment_1"
                      ]

# Add missing features with default value 0
for f in expected_features:
    if f not in X_train.columns:
        X_train[f] = 0
    if f not in X_val.columns:
        X_val[f] = 0

# Keep only expected features
X_train = X_train[expected_features]
X_val = X_val[expected_features]

# Initialize and train Random Forest model
print("Initializing and training random forest model...")
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train["future_return_30m"].values.ravel())
print(model.feature_names_in_)

# Intialize and train Random Forest Classifier model
# print("Initializing and training random forest classifier model...")
# model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train["target_direction"])

# Save feature names inside the model for deployment
model.feature_names_in_ = expected_features

# Save the trained model
joblib.dump(model, f"{PATH_BARS}/news model/random_forest_model_news.pkl")
print("Model saved successfully.")

# Deviation per symbol
print("Plotting results...")
results_per_symbol = []

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, symbol in enumerate(SYMBOLS):
    ax = axes[i]

    # Filter data for symbol
    mask = y_val["symbol"] == symbol
    y_val_sym = y_val.loc[mask, "future_return_30m"]
    X_val_sym = X_val.loc[mask]

    # Model predictions
    pred_sym = model.predict(X_val_sym)

    # Baseline: mean return per symbol
    baseline_return_sym = y_val_sym.mean()
    pred_baseline_sym = [baseline_return_sym] * len(y_val_sym)

    # Calculate MAE
    mae_model_sym = mean_absolute_error(y_val_sym, pred_sym)
    mae_baseline_sym = mean_absolute_error(y_val_sym, pred_baseline_sym)

    results_per_symbol.append((symbol, mae_model_sym, mae_baseline_sym))

    # Calculate absolute deviation
    abs_dev_model = abs(y_val_sym.values - pred_sym)
    abs_dev_baseline = abs(y_val_sym.values - pred_baseline_sym)

    # Plot for this symbol
    ax.plot(y_val_sym.index, abs_dev_model, label="Model Deviation")
    ax.plot(y_val_sym.index, abs_dev_baseline, label="Baseline Deviation", linestyle="--")
    ax.set_title(f"{symbol}\nMAE Model={mae_model_sym:.4f}, MAE Baseline={mae_baseline_sym:.4f}")
    ax.set_ylabel("Absolute Deviation")
    ax.set_xlabel("Timestamp")
    ax.legend()

plt.suptitle("Absolute Deviations: Model vs. Baseline per Symbol", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{PATH_FIGURE}/07_random_forest_MAE_per_symbol.png");plt.close()

print("Saving results...")
# Save MAE results per symbol
results_df = pd.DataFrame(results_per_symbol, columns=["Symbol", "MAE_Model", "MAE_Baseline"])
print(results_df)
results_df.to_csv(f"{PATH_BARS}/news_random_forest_train_metrics_per_symbol.csv", index=False)