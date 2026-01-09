import yaml
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Load parameters
params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
SYMBOLS = params['DATA_ACQUISITON']['SYMBOLS']
PATH_FIGURE = params['DATA_UNDERSTANDING']['FIGURE_PATH']

# Load test set
X_test = pd.read_parquet(f"{PATH_BARS}/news model/X_test_news.parquet")
y_test = pd.read_parquet(f"{PATH_BARS}/news model/y_test_news.parquet")

# Load model
model = joblib.load(f"{PATH_BARS}/news model/random_forest_model_news.pkl")

# Align X_test to model features
expected = list(model.feature_names_in_)
X_test = X_test[[f for f in expected if f in X_test.columns]]
for f in expected:
    if f not in X_test.columns:
        X_test[f] = 0
X_test = X_test[expected]

# Backtest per symbol
results_per_symbol = []
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, symbol in enumerate(SYMBOLS):
    ax = axes[i]

    # Filter test data for symbol
    mask = y_test["symbol"] == symbol
    y_test_sym = y_test.loc[mask, "future_return_30m"]
    X_test_sym = X_test.loc[mask]

    # Align X_test_sym to model features
    X_test_sym = X_test_sym[expected]

    # Model predictions
    pred_sym = model.predict(X_test_sym)

    # Trading rule: Long if prediction > 0
    signals = (pred_sym > 0).astype(int)
    strategy_returns = signals * y_test_sym.values

    # Performance metrics
    cum_strategy = np.cumsum(strategy_returns)
    cum_market = np.cumsum(y_test_sym.values)
    total_strategy = cum_strategy[-1]
    total_market = cum_market[-1]
    win_rate = (strategy_returns > 0).mean()

    results_per_symbol.append((symbol, total_strategy, total_market, win_rate))

    # Plot performance
    ax.plot(y_test_sym.index, cum_market, label="Market")
    ax.plot(y_test_sym.index, cum_strategy, label="Strategy")
    ax.set_title(f"{symbol}\nStrategy={total_strategy:.4f}, Market={total_market:.4f}, WinRate={win_rate:.2%}")
    ax.legend()

plt.suptitle("Backtest Performance per Symbol (Test Set)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{PATH_FIGURE}/09_backtest_testset_per_symbol.png"); plt.close()

# Save results
results_df = pd.DataFrame(results_per_symbol, columns=["Symbol", "Total_Strategy", "Total_Market", "WinRate"])
results_df.to_csv(f"{PATH_BARS}/news_random_forest_backtest_testset_per_symbol.csv", index=False)
