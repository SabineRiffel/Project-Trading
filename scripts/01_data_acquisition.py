from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
from datetime import datetime
import pytz
import yaml

# Download stock bar data from Alpaca API and save as Parquet file
# Load API credential from YAML configuration file
print('Loading API keys...')
keys = yaml.safe_load(open("../data/conf/keys.yaml"))
API_KEY = keys['KEYS']['APCA-API-KEY-ID-Data']
SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY-Data']

# Load data acquisition parameters from YAML configuration file
print('Loading parameters...')
params = yaml.safe_load(open("../data/conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
START_DATE = datetime.strptime(params['DATA_ACQUISITON']['START_DATE'], "%Y-%m-%d")
END_DATE = datetime.strptime(params['DATA_ACQUISITON']['END_DATE'], "%Y-%m-%d")
SYMBOLS = params['DATA_ACQUISITON']['SYMBOLS']

# Initialize the Alpaca client with API credentials
print('Initializing Alpaca client...')
client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY)

# Get market calendar for in params.yaml file specified date range
print('Fetching market calendar...')
cal_request = GetCalendarRequest(start=START_DATE, end=END_DATE)
calendar = trading_client.get_calendar(cal_request)

# Build lookup table (date → open_dt, close_dt)
print('Building market hours lookup table...')
cal_map = {}
eastern = pytz.timezone("US/Eastern")

print('Processing calendar entries...')
# Populate the calendar map with localized open and close times
for c in calendar:
    open_dt = eastern.localize(c.open)
    close_dt = eastern.localize(c.close)
    cal_map[c.date] = (open_dt, close_dt)

# Add market open flag
print('Defining market open checker function...')
def check_open(ts):
    ts_eastern = ts.tz_convert(eastern) if ts.tzinfo else ts.tz_localize("UTC").astimezone(eastern)
    d = ts_eastern.date()
    if d not in cal_map:
        return False
    open_dt, close_dt = cal_map[d]
    return open_dt <= ts_eastern < close_dt

# Create a request object for historical bar data
print(f'Creating data request for {SYMBOLS} from {START_DATE} to {END_DATE}...')
request = StockBarsRequest(
    symbol_or_symbols=SYMBOLS[0],
    timeframe=TimeFrame.Minute,
    adjustment=Adjustment.ALL,
    start=START_DATE,
    end=END_DATE
)

# Retrieve bar data from Alpaca API
print('Downloading bar data...')
bars = client.get_stock_bars(request)
df = bars.df
df.reset_index(inplace=True)

# Remove the 'symbol' column if it exists
if 'symbol' in df.columns:
    df.drop(columns=['symbol'], inplace=True)

# Add a column to indicate if the market was open at the timestamp
df["is_open"] = df["timestamp"].map(check_open)

# Filter the DataFrame to include only rows where the market was open
df = df[df['is_open']]

# Drop the 'is_open' column as it's no longer needed
df.drop(columns=['is_open'], inplace=True)

# Save the DataFrame as a Parquet file for efficient storage
print(f'Saving data to {PATH_BARS}/{SYMBOLS}.parquet...')
#df.to_parquet(f'{PATH_BARS}/{SYMBOLS[0]}.parquet', index=False)
print('Data acquisition complete.')