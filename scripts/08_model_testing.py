import yaml
import pandas as pd
import matplotlib.pyplot as plt

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
import joblib
model = joblib.load(f"{PATH_BARS}/news model/random_forest_model_news.pkl")

# Evaluate model on test set
test_score = model.score(X_test, y_test)
test_pred_rf = model.predict(X_test)

# Save test metrics to a csv file
test_metrics = pd.DataFrame({"Metric": ["R^2 Score"],"Value": [test_score]})
test_metrics.to_csv(f"{PATH_BARS}/news model/random_forest_test_metrics_news.csv", index=False)

# Plot test results
plt.figure(figsize=(10, 4), dpi=120)
plt.title("Test Results of Random Forest Model")
plt.plot(y_test.index, y_test.values, label="Actual")
plt.plot(y_test.index, test_pred_rf, label="Predicted")
plt.ylabel("Future Return 30m")
plt.xlabel("Timestamp")
plt.tight_layout()
plt.legend()
plt.savefig(f"{PATH_FIGURE}/08_random_forest_test_news.png");plt.close()

