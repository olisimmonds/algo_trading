import pandas as pd
from datetime import datetime, timedelta
from core_code.get_fin_info import get_s_p_tickers
from core_code.get_fin_info import get_fin_data
import pickle
import random


def sort_fin_data(ticker, start, end, interval):
    data = get_fin_data(ticker, start, end, interval)
    times = data.index.tolist()
    prices = data["Close"].to_numpy().flatten().tolist()
    volumes = data["Volume"].to_numpy().flatten().tolist()
    return pd.DataFrame({"Time": times, "Close": prices, "Volume": volumes})

# Parameters
n_days = 35
tickers = get_s_p_tickers()
random.shuffle(tickers)
tickers = tickers[:50]

# Data Storage
with open('data_records.pkl', 'rb') as file:
    data_records = pickle.load(file)
# I have run this once as of 14/02/25

for i in range(n_days):
    date = datetime.today() - timedelta(days=n_days - i)
    start_date = (date).strftime('%Y-%m-%d')
    day_data = {}
    
    # Only intrested in weekdays
    if date.weekday() < 5:
        for ticker in tickers:
            interval = "5m"
            end = (datetime.today() - timedelta(days=n_days - i - 1)).strftime('%Y-%m-%d')
            
            stock_data = sort_fin_data(ticker, start_date, end, interval)
            day_data[ticker] = stock_data
            
    data_records[start_date] = day_data

with open('data_records.pkl', 'wb') as file:
    pickle.dump(data_records, file)
