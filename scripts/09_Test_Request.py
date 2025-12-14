import requests

url = "http://127.0.0.1:8000/predict"
payload = {
        "features": {
            "ema_5": 123.4,
            "ema_10": 125.6,
            "ema_15": 127.8,
            "ema_30": 130.0,
            "ema_5_slope": 0.01,
            "ema_10_slope": 0.01,
            "ema_15_slope": 0.02,
            "ema_30_slope": 0.03,
            "ema_5_accel": 0.02,
            "ema_10_accel": 0.02,
            "ema_15_accel": 0.03,
            "ema_30_accel": 0.04,
            "close": 128.5,
            "volume": 100000,
            "vwap": 129.0,
            "sentiment_-1": 0,
            "sentiment_0": 1,
            "sentiment_1": 0
    }
}

response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
print("Response Text:", response.text)
