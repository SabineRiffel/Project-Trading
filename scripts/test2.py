

import pandas as pd

df = pd.read_csv("/Users/sabine/PycharmProjects/Project-Trading/data/AAPL_news.csv")

df.describe(include="object").to_csv("/Users/sabine/PycharmProjects/Project-Trading/data/descrie.csv", index=True)
#df.describe().to_csv("/Users/sabine/PycharmProjects/Project-Trading/data/descrie.csv", index=True)
