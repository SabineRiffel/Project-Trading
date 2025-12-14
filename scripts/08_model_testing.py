import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
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
X_test = pd.read_parquet(f"{PATH_BARS}/news model/X_test_news.parquet")
y_test = pd.read_parquet(f"{PATH_BARS}/news model/y_test_news.parquet")

# Load trained model
model = joblib.load(f"{PATH_BARS}/news model/random_forest_model_news.pkl")

# Deviation per symbol
results_per_symbol = []
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, symbol in enumerate(SYMBOLS):
    ax = axes[i]
    mask = y_test["symbol"] == symbol
    y_test_sym = y_test.loc[mask, "future_return_30m"]
    X_test_sym = X_test.loc[mask]

    # Predictions
    pred_sym = model.predict(X_test_sym)

    # Baseline: mean return per symbol
    baseline_return_sym = y_test_sym.mean()
    pred_baseline_sym = [baseline_return_sym] * len(y_test_sym)

    # MAE calculation
    mae_model_sym = mean_absolute_error(y_test_sym, pred_sym)
    mae_baseline_sym = mean_absolute_error(y_test_sym, pred_baseline_sym)
    results_per_symbol.append((symbol, mae_model_sym, mae_baseline_sym))

    # Absolute deviations
    abs_dev_model = abs(y_test_sym.values - pred_sym)
    abs_dev_baseline = abs(y_test_sym.values - pred_baseline_sym)

    # Plot for this symbol
    ax.plot(y_test_sym.index, abs_dev_model, label="Model Deviation")
    ax.plot(y_test_sym.index, abs_dev_baseline, label="Baseline Deviation", linestyle="--")
    ax.set_title(f"{symbol}\nMAE Model={mae_model_sym:.4f}, MAE Baseline={mae_baseline_sym:.4f}")
    ax.set_ylabel("Absolute Deviation")
    ax.set_xlabel("Timestamp")
    ax.legend()

plt.suptitle("Test: Absolute Deviations per Symbol", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{PATH_FIGURE}/08_random_forest_test_per_symbol.png")
plt.close()

# Save MAE results per symbol
results_df = pd.DataFrame(results_per_symbol, columns=["Symbol", "MAE_Model", "MAE_Baseline"])
print(results_df)
results_df.to_csv(f"{PATH_BARS}/news model/random_forest_test_metrics_per_symbol.csv", index=False)
