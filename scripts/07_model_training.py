import yaml
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
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

# Initialize and train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train.values.ravel())

# Evaluate model on validation set
val_score = model.score(X_val, y_val)
val_pred_rf = model.predict(X_val)

print(val_score)
print(val_pred_rf)


# Save the trained model
joblib.dump(model, f"{PATH_BARS}/news model/random_forest_model_news.pkl")

# Plot validation results
plt.figure(figsize=(10, 4), dpi=120)
plt.title("Validation Results of Random Forest Model")
plt.plot(y_val.index, y_val.values, label="Actual")
plt.plot(y_val.index, val_pred_rf, label="Predicted")
plt.ylabel("Future Return 30m")
plt.xlabel("Timestamp")
plt.tight_layout()
plt.legend()
plt.savefig(f"{PATH_FIGURE}/07_random_forest_validation_news.png");plt.close()


