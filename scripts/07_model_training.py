import yaml
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Load parameters
params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
SYMBOLS = params['DATA_ACQUISITON']['SYMBOLS']
PATH_FIGURE = params['DATA_UNDERSTANDING']['FIGURE_PATH']

# Load training, validation
X_train = pd.read_parquet(f"{PATH_BARS}/news model/X_train_news.parquet")
y_train = pd.read_parquet(f"{PATH_BARS}/news model/y_train_news.parquet")
X_val = pd.read_parquet(f"{PATH_BARS}/news model/X_val_news.parquet")
y_val = pd.read_parquet(f"{PATH_BARS}/news model/y_val_news.parquet")

# Initialize and train Random Forest model
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train["future_return_30m"].values.ravel())

# Save the trained model
joblib.dump(model, f"{PATH_BARS}/news model/random_forest_model_news.pkl")

# Deviation per symbol
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