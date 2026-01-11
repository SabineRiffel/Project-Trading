import time
import schedule
from alpaca_trade_api.rest import REST
import pandas as pd
import joblib

# Configuration (load from a config file in practice)
API_KEY = "your_paper_key"
SECRET_KEY = "your_paper_secret"
BASE_URL = "https://paper-api.alpaca.markets"
MODEL_PATH = "models/senator_randomforest.pkl"

# Initialize
api = REST(API_KEY, SECRET_KEY, BASE_URL)
model = joblib.load(MODEL_PATH)


def fetch_recent_filings_lookback(hours=24):
    """
    In a real scenario, you would query an API or database
    for recent senator filings from the past `hours` hours.
    This is a placeholder for your data acquisition logic.
    """
    # Example: Read from your updated test set or a live feed
    # This should return a DataFrame with 'Ticker', 'TimeOfFiled', 'signed_amount', etc.
    pass


def check_and_trade():
    print(f"[{pd.Timestamp.now()}] Checking for new signals...")

    # 1. Get recent filings
    df_new = fetch_recent_filings_lookback(hours=6)

    if df_new.empty:
        print("No new filings.")
        return

    # 2. For each filing, get recent market data and build features
    for idx, row in df_new.iterrows():
        try:
            # Get recent bars (like in reference code)
            bars = api.get_bars(row['Ticker'], "15Min", limit=20).df
            if bars.empty:
                continue

            # Build your feature vector for the model
            # ... (Your feature engineering logic here) ...
            features = pd.DataFrame([your_features])

            # Ensure feature order matches model
            features = features[model.feature_names_in_]

            # Predict
            pred = model.predict(features)[0]

            # Trading logic
            current_pos = get_position(row['Ticker'])
            if pred > THRESHOLD and not current_pos:
                submit_order(row['Ticker'], qty=10, side='buy')
                log_trade(row['Ticker'], 'BUY', pred)
            elif pred < EXIT_THRESHOLD and current_pos:
                submit_order(row['Ticker'], qty=current_pos, side='sell')
                log_trade(row['Ticker'], 'SELL', pred)

        except Exception as e:
            print(f"Error processing {row['Ticker']}: {e}")
            continue


def get_position(symbol):
    """Check existing position for a symbol."""
    try:
        pos = api.get_position(symbol)
        return float(pos.qty)
    except:
        return 0


def submit_order(symbol, qty, side):
    """Submit a paper trade order."""
    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        print(f"Paper {side.upper()} order for {qty} {symbol}")
    except Exception as e:
        print(f"Order failed: {e}")


# Schedule to run every 30 minutes
schedule.every(30).minutes.do(check_and_trade)

if __name__ == "__main__":
    print("Starting Senator-Based Paper Trading Bot...")
    while True:
        schedule.run_pending()
        time.sleep(60)