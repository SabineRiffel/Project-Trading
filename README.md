# Forecasting Apple Stock Prices
##### Author: Laura Wozniak (580635) and Sabine Riffel (582950)


## Problem Definition
**Target**

The goal of this project is to predict the intraday trend direction of Apple Inc. (AAPL) stock prices over the next t = [5, 10, 15, 30, 60] minutes using historical minute-level data from 2022-01-01 to 2025-06-30. 
We use a Random Forest model with features including the linear regression slope over each t-minute window, normalized by the mean price, to classify the trend direction.

**Input Features**

Normalized VWAP (volume weighted average price) and volume
Normalized exponential moving average (EMA) over t = [5, 10, 15, 30, 60] minutes
Linear regression slope of EMAs over t = [5, 10, 15, 30, 60] minutes
Second-order slope (acceleration) of EMAs over t = [5, 10, 15, 30, 60] minutes

---

## Table of Contents

- [1 - Data Acquisition](#1-data-acquisition)

---

## 1 - Data Acquisition

**Description**

Retrieves historical stock price data for Apple Inc. (AAPL) using the Alpaca API. The dataset contains minute-level data from **2022-01-01 to 2025-06-30**.
The data request specified the ticker symbol AAPL and included fields such as timestamp, open, high, low, close, volume, trade_count, and vwap. 
The retrieved data is stored as a Parquet file named `AAPL.parquet` in the `../data/` directory for efficient storage and fast retrieval.

**Script**

[01_data_acquisition.py](scripts/01_data_acquisition.py)

**Data**

![data.png](images/01_raw_data.png)

**Columns**
- `timestamp`: Date and time of the stock price record
- `open`: Opening price of the stock at the given timestamp
- `high`: Highest price of the stock at the given timestamp
- `low`: Lowest price of the stock at the given timestamp
- `close`: Closing price of the stock at the given timestamp
- `volume`: Number of shares traded during the given timestamp
- `trade_count`: Number of trades executed during the given timestamp'
- `vwap`: Volume Weighted Average Price during the given timestamp

---

## 2 - Data Understanding
## 3 - Pre-Split Preparation
## 4 - Split Data
## 5 - Post-Split Preparation
## 6 - Feature Selection
## 7 - Model Training & Validation
## 8 - Final Testing
## 9 - Deployment



