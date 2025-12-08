import yaml
import pandas as pd
import matplotlib.pyplot as plt

# Load parameters
params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
SYMBOLS = params['DATA_ACQUISITON']['SYMBOLS']
PATH_FIGURE = params['DATA_UNDERSTANDING']['FIGURE_PATH']

# Load training features and target variable
X_train = pd.read_parquet(f"{PATH_BARS}/news model/X_train_news.parquet")
y_train = pd.read_parquet(f"{PATH_BARS}/news model/y_train_news.parquet")

# Feature selection using correlation with the target variable
correlations = X_train.join(y_train).corr()

# Remove rows and columns with all NaN values
correlations = correlations.dropna(how="all").dropna(axis=1, how="all")

# Plot correlation matrix
plt.figure(figsize=(10,8))
plt.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(correlations.columns)), correlations.columns, rotation=90)
plt.yticks(range(len(correlations.index)), correlations.index)
plt.title("Correlation Matrix of Features and Target Variable")
plt.tight_layout()
plt.savefig(f"{PATH_FIGURE}/06_correlation_matrix_news.png");plt.close()
