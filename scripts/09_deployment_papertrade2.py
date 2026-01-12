import os
import time
import yaml
import joblib
import datetime as dt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import alpaca_trade_api as tradeapi
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI

# Load API credential from YAML configuration file
print("Loading configuration...")
keys = yaml.safe_load(open("../data/conf/keys.yaml"))
#keys = yaml.safe_load(open("/conf/keys.yaml"))
API_KEY = keys["KEYS"]["APCA-API-KEY-ID-Data-2"]
SECRET_KEY = keys["KEYS"]["APCA-API-SECRET-KEY-Data-2"]
BASE_URL = keys["KEYS"]["APCA-BASE-URL-Data-2"]

params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
SYMBOLS = params["DATA_ACQUISITON"]["SYMBOLS"]
PATH_FIGURE = params['DATA_UNDERSTANDING']['FIGURE_PATH']

# Load trained model
model = joblib.load(f"{PATH_BARS}/news model/random_forest_model_news.pkl")
print(model.feature_names_in_)
#model = joblib.load("data/news model/random_forest_model_news.pkl")
app = FastAPI(title="Trading API", version="1.0")
# Trading controls
QTY_PER_TRADE = params.get("DEPLOYMENT", {}).get("QTY_PER_TRADE", 2)
INTERVAL_SECONDS = params.get("DEPLOYMENT", {}).get("INTERVAL_SECONDS", 60)
BAR_LIMIT_FOR_FEATURES = params.get("DEPLOYMENT", {}).get("BAR_LIMIT_FOR_FEATURES", 30)

# Initialize the Alpaca client with API credentials
print("Initializing Alpaca clients...")
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
news_client = NewsClient(API_KEY, SECRET_KEY)

print("Initializing sentiment analyzer...")
analyzer = SentimentIntensityAnalyzer()

# Feature engineering
def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def build_features_from_bars(df_bars: pd.DataFrame, sentiment_label: int) -> pd.DataFrame:
    if "close" not in df_bars.columns or "volume" not in df_bars.columns:
        raise ValueError("df_bars must contain 'close' and 'volume' columns.")
    if "vwap" not in df_bars.columns or df_bars["vwap"].isna().all():
        df_bars["vwap"] = df_bars["close"]

    close = df_bars["close"]

    # EMAs
    ema_5_series = compute_ema(close, 5)
    ema_10_series = compute_ema(close, 10)
    ema_15_series = compute_ema(close, 15)
    ema_30_series = compute_ema(close, 30)

    # Current EMA values
    ema_5 = ema_5_series.iloc[-1]
    ema_10 = ema_10_series.iloc[-1]
    ema_15 = ema_15_series.iloc[-1]
    ema_30 = ema_30_series.iloc[-1]

    # Slopes (first difference of EMA series)
    ema_5_slope = ema_5_series.diff().iloc[-1]
    ema_10_slope = ema_10_series.diff().iloc[-1]
    ema_15_slope = ema_15_series.diff().iloc[-1]
    ema_30_slope = ema_30_series.diff().iloc[-1]

    # Accelerations (second difference of EMA series)
    ema_5_accel = ema_5_series.diff().diff().iloc[-1]
    ema_10_accel = ema_10_series.diff().diff().iloc[-1]
    ema_15_accel = ema_15_series.diff().diff().iloc[-1]
    ema_30_accel = ema_30_series.diff().diff().iloc[-1]

    # Volume Spike
    roll_vol = df_bars["volume"].rolling(30, min_periods=3).mean()
    last_roll = roll_vol.iloc[-1]
    if pd.isna(last_roll) or last_roll == 0:
        volume_spike = 1.0
    else:
        volume_spike = df_bars["volume"].iloc[-1] / last_roll

    # One-hot sentiment mapping to three features
    s_neg = 1 if sentiment_label == -1 else 0
    s_neu = 1 if sentiment_label == 0 else 0
    s_pos = 1 if sentiment_label == 1 else 0

    # Build single-row DataFrame with exact feature names expected by the model
    features = pd.DataFrame([{
        "ema_5": float(ema_5),
        "ema_10": float(ema_10),
        "ema_15": float(ema_15),
        "ema_30": float(ema_30),
        "ema_5_slope": float(ema_5_slope),
        "ema_10_slope": float(ema_10_slope),
        "ema_15_slope": float(ema_15_slope),
        "ema_30_slope": float(ema_30_slope),
        "ema_5_accel": float(ema_5_accel),
        "ema_10_accel": float(ema_10_accel),
        "ema_15_accel": float(ema_15_accel),
        "ema_30_accel": float(ema_30_accel),
        "close": float(df_bars["close"].iloc[-1]),
        "volume": float(df_bars["volume"].iloc[-1]),
        "vwap": float(df_bars["vwap"].iloc[-1]),
        "volume_spike": float(volume_spike),
        "sentiment_-1": int(s_neg),
        "sentiment_0": int(s_neu),
        "sentiment_1": int(s_pos),
    }])

    return features

# Fetch latest news for symbol and compute VADER sentiment
def latest_headline_sentiment(symbol: str) -> int:
    try:
        req = NewsRequest(symbol_or_symbols=[symbol], limit=1)
        news_set = news_client.get_news(req)
        news_list = list(news_set)
        if not news_list:
            return 0
        news_item = news_list[0][0]

        text = None
        if hasattr(news_item, "headline") and news_item.headline:
            text = news_item.headline
        elif hasattr(news_item, "summary") and news_item.summary:
            text = news_item.summary

        if not text:
            return 0

        score = analyzer.polarity_scores(text)["compound"]
        if score > 0.05:
            return 1
        elif score < -0.05:
            return -1
        else:
            return 0
    except Exception as e:
        print(f"[WARN] Sentiment fetch failed for {symbol}: {e}")
        return 0

# Fetch latest minute bars for asymbol and return a dataframe with columns
def get_latest_bars(api: tradeapi.REST, symbol: str, limit: int = BAR_LIMIT_FOR_FEATURES) -> pd.DataFrame:
    bars = api.get_bars(symbol, "1Min", limit=limit)
    if not bars:
        return pd.DataFrame()
    raw = []
    for b in bars:
        raw.append({
            "timestamp": pd.to_datetime(b.t, utc=True),
            "close": float(b.c),
            "volume": float(b.v),
            "vwap": float(b.vw) if hasattr(b, "vw") and b.vw is not None else float(b.c),
        })
    df = pd.DataFrame(raw).sort_values("timestamp")
    return df

results = []

def log_event(timestamp, symbol, signal, price, pred, sentiment_label):
    results.append({
        "timestamp": timestamp,
        "symbol": symbol,
        "signal": signal,
        "price": price,
        "predicted_return": pred,
        "sentiment_label": sentiment_label,
    })

def flush_logs(path: str = "../data/paper_trading_log.csv"):
    if not results:
        return
    df = pd.DataFrame(results)
    mode = "a" if os.path.exists(path) else "w"
    header = not os.path.exists(path)
    df.to_csv(path, index=False, mode=mode, header=header)
    results.clear()
    print(f"[INFO] Flushed logs to {path}")

# Paper trading loop
def main():
    print("Starting Paper Trading...")
    iteration = 0
    while True:
        now = dt.datetime.now(dt.timezone.utc)

        for symbol in SYMBOLS:
            try:
                # Fetch latest bars
                df_bars = get_latest_bars(api, symbol, limit=BAR_LIMIT_FOR_FEATURES)
                if df_bars.empty or len(df_bars) < 5:
                    print(f"[WARN] Not enough bars for {symbol} at {now}. Skipping.")
                    continue

                # Fetch latest sentiment via news
                sentiment_label = latest_headline_sentiment(symbol)

                # Build features
                X_live = build_features_from_bars(df_bars, sentiment_label)

                # Adjust features to match model's expected input
                expected = list(model.feature_names_in_)
                # Remove unknown features
                X_live = X_live[[f for f in expected if f in X_live.columns]]
                # Add missing features with default 0
                for f in expected:
                    if f not in X_live.columns:
                        X_live[f] = 0

                # Reorder columns
                X_live = X_live[expected]

                # Predict
                pred = float(model.predict(X_live)[0])

                # Derive signal
                if pred > 0:
                    signal = "LONG"
                else:
                    signal = "FLAT"

                # Optionally place paper order (commented safeguard)
                if signal == "LONG":
                    try:
                        api.submit_order(
                            symbol=symbol,
                            qty=QTY_PER_TRADE,
                            side="buy",
                            type="market",
                            time_in_force="gtc"
                        )
                        print(f"[ORDER] LONG {symbol} qty={QTY_PER_TRADE} at ~{df_bars['close'].iloc[-1]:.2f}")
                    except Exception as oe:
                        print(f"[WARN] Order submit failed for {symbol}: {oe}")
                else:
                    print(f"[INFO] No action for {symbol} at {now} with signal={signal}")

                # Log event
                log_event(
                    timestamp=now.isoformat(),
                    symbol=symbol,
                    signal=signal,
                    price=float(df_bars["close"].iloc[-1]),
                    pred=pred,
                    sentiment_label=sentiment_label,
                )

                print(f"{now} | {symbol} | signal={signal} | pred={pred:.5f} | "
                      f"price={df_bars['close'].iloc[-1]:.2f} | sent={sentiment_label}")

            except Exception as e:
                print(f"[ERROR] Loop error for {symbol} at {now}: {e}")

        iteration += 1
        # Flush logs every 10 iterations to avoid memory growth
        if iteration % 10 == 0:
            flush_logs()

        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping Paper Trading...")
        flush_logs()

# Load paper trading log
df = pd.read_csv("../data/paper_trading_log2.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
