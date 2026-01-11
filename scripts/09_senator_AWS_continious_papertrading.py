import os
import time
import yaml
import joblib
import datetime as dt
import pandas as pd
import alpaca_trade_api as tradeapi
import schedule

# =====================
# Config
# =====================
keys = yaml.safe_load(open("../data/conf/keys.yaml"))
params = yaml.safe_load(open("../data/conf/params.yaml"))

API_KEY = keys["KEYS"]["APCA-API-KEY-ID-Data"]
SECRET_KEY = keys["KEYS"]["APCA-API-SECRET-KEY-Data"]
BASE_URL = "https://paper-api.alpaca.markets"

SYMBOLS = params["DATA_ACQUISITON"]["SYMBOLS"]
QTY_PER_TRADE = params["DEPLOYMENT"]["QTY_PER_TRADE"]
BAR_LIMIT = params["DEPLOYMENT"]["BAR_LIMIT_FOR_FEATURES"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "senator_randomforest.pkl")

LOG_FILE = "../data/paper_trading_log.csv"

CONF_THRESHOLD = 0.01  # adjust based on backtest

# =====================
# Init
# =====================
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)

# =====================
# Helpers
# =====================
def get_latest_bars(symbol):
    bars = api.get_bars(symbol, "1Min", limit=BAR_LIMIT)
    data = [{
        "close": b.c,
        "volume": b.v,
        "vwap": b.vw if b.vw else b.c
    } for b in bars]
    return pd.DataFrame(data)

def has_position(symbol):
    try:
        pos = api.get_position(symbol)
        return float(pos.qty) > 0
    except:
        return False

def market_is_open():
    try:
        clock = api.get_clock()
        return clock.is_open
    except:
        return False

def log_trade(row):
    df = pd.DataFrame([row])
    mode = "a" if os.path.exists(LOG_FILE) else "w"
    df.to_csv(LOG_FILE, mode=mode, header=not os.path.exists(LOG_FILE), index=False)

# =====================
# Trading Function
# =====================
def trade():
    now = dt.datetime.utcnow()

    if not market_is_open():
        print(f"{now} | Market closed, skipping trade.")
        return

    for symbol in SYMBOLS:
        try:
            df_bars = get_latest_bars(symbol)
            if df_bars.empty or len(df_bars) < 5:
                continue

            X = df_bars.iloc[-1:][FEATURES]
            pred = float(model.predict(X)[0])

            signal = "LONG" if pred > CONF_THRESHOLD else "FLAT"
            in_position = has_position(symbol)

            if signal == "LONG" and not in_position:
                api.submit_order(
                    symbol=symbol,
                    qty=QTY_PER_TRADE,
                    side="buy",
                    type="market",
                    time_in_force="gtc"
                )

            account = api.get_account()

            log_trade({
                "timestamp": now,
                "symbol": symbol,
                "signal": signal,
                "price": float(df_bars["close"].iloc[-1]),
                "prediction": pred,
                "equity": float(account.equity)
            })

            print(f"{now} | {symbol} | {signal} | pred={pred:.5f} | equity={account.equity}")

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")

# =====================
# Scheduler Setup
# =====================
print("🚀 Starting Alpaca Paper Trading Scheduler")

# Run trade every minute
schedule.every(1).minutes.do(trade)

while True:
    schedule.run_pending()
    time.sleep(1)
