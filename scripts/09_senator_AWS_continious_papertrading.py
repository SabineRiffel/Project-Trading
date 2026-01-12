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

# Paper accounts (3 separate accounts)
ACCOUNTS = [
    {
        "name": "PA3PVPIP58NB",
        "api_key": keys["KEYS"]["APCA-API-KEY-ID-Data"],
        "secret_key": keys["KEYS"]["APCA-API-SECRET-KEY-Data"],
        "conf_threshold": 0.01 #Default/backtested value
    },
    {
        "name": "PA38NUSB8Y1K",
        "api_key": keys["KEYS"]["APCA-API-KEY-ID-Data-2"],
        "secret_key": keys["KEYS"]["APCA-API-SECRET-KEY-Data-2"],
        "conf_threshold": 0.02 #Fewer trades, only strongest signals, more conservative
    },
    {
        "name": "PA39DG0DV36M",
        "api_key": keys["KEYS"]["APCA-API-KEY-ID-Data-3"],
        "secret_key": keys["KEYS"]["APCA-API-SECRET-KEY-Data-3"],
        "conf_threshold": 0.005 #More trades, possibly riskier, can catch small signals
    },
]

BASE_URL = "https://paper-api.alpaca.markets"
SYMBOLS = params["DATA_ACQUISITON"]["SYMBOLS"]
QTY_PER_TRADE = params["DEPLOYMENT"]["QTY_PER_TRADE"]
BAR_LIMIT = params["DEPLOYMENT"]["BAR_LIMIT_FOR_FEATURES"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "senator_randomforest.pkl")
LOG_FILE = "../data/paper_trading_log.csv"

# =====================
# Init
# =====================
# Load the model once
model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)

# Create API clients for each account
for acc in ACCOUNTS:
    acc["api"] = tradeapi.REST(acc["api_key"], acc["secret_key"], BASE_URL, api_version="v2")

# =====================
# Helpers
# =====================
def get_latest_bars(api, symbol):
    bars = api.get_bars(symbol, "1Min", limit=BAR_LIMIT)
    data = [{"close": b.c, "volume": b.v, "vwap": b.vw if b.vw else b.c} for b in bars]
    return pd.DataFrame(data)

def has_position(api, symbol):
    try:
        pos = api.get_position(symbol)
        return float(pos.qty) > 0
    except:
        return False

def market_is_open(api):
    try:
        clock = api.get_clock()
        return clock.is_open
    except:
        return False

def log_trade(row):
    df = pd.DataFrame([row])
    mode = "a" if os.path.exists(LOG_FILE) else "w"
    df.to_csv(LOG_FILE, mode=mode, header=not os.path.exists(LOG_FILE), index=False)

def build_features(df, now):
    # Use last two bars to recreate training features
    last = df.iloc[-1]
    prev = df.iloc[-2]

    features = {
        "signed_amount": (last["close"] - prev["close"]) * last["volume"],
        "tx_hour": now.hour,
        "tx_weekday": now.weekday(),
        "price_before": prev["close"],
        "vol_before": prev["volume"],
        "vwap_before": prev["vwap"],
    }

    return pd.DataFrame([features])

# =====================
# Trading Function
# =====================
def trade():
    now = dt.datetime.utcnow()

    # Check market once (assuming all accounts use the same market)
    if not market_is_open(ACCOUNTS[0]["api"]):
        print(f"{now} | Market closed, skipping all trades.")
        return

    for acc in ACCOUNTS:
        api = acc["api"]

        for symbol in SYMBOLS:
            try:
                df_bars = get_latest_bars(api, symbol)
                if df_bars.empty or len(df_bars) < 5:
                    continue

                X = build_features(df_bars, now)[FEATURES]
                pred = float(model.predict(X)[0])
                signal = "LONG" if pred > acc["conf_threshold"] else "FLAT"
                in_position = has_position(api, symbol)

                if signal == "LONG" and not in_position:
                    api.submit_order(
                        symbol=symbol,
                        qty=QTY_PER_TRADE,
                        side="buy",
                        type="market",
                        time_in_force="gtc"
                    )

                account_info = api.get_account()
                log_trade({
                    "timestamp": now,
                    "account": acc["name"],
                    "symbol": symbol,
                    "signal": signal,
                    "price": float(df_bars["close"].iloc[-1]),
                    "prediction": pred,
                    "equity": float(account_info.equity)
                })

                print(f"{now} | {acc['name']} | {symbol} | {signal} | pred={pred:.5f} | equity={account_info.equity}")

            except Exception as e:
                print(f"[ERROR] {acc['name']} | {symbol}: {e}")

# =====================
# Scheduler Setup
# =====================
print("🚀 Starting Alpaca Paper Trading Scheduler (3 Accounts)")

schedule.every(1).minutes.do(trade)

while True:
    schedule.run_pending()
    time.sleep(1)
