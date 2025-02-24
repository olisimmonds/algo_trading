import pandas as pd
from datetime import datetime, timedelta
from src.core_code.get_fin_info import get_s_p_tickers
from src.core_code.get_fin_info import get_fin_data
import pickle


def sort_fin_data(ticker, start, end, interval):
    data = get_fin_data(ticker, start, end, interval)
    times = data.index.tolist()
    close_prices = data["Close"].to_numpy().flatten().tolist()
    high_prices = data["High"].to_numpy().flatten().tolist()
    low_prices = data["Low"].to_numpy().flatten().tolist()
    open_prices = data["Open"].to_numpy().flatten().tolist()
    volumes = data["Volume"].to_numpy().flatten().tolist()
    return pd.DataFrame({"Ticker": ticker, "Time": times, "Close": close_prices, "High": high_prices, "Low": low_prices, "Open": open_prices, "Volume": volumes})

# Parameters
tickers = get_s_p_tickers()
tickers = tickers

date = datetime.today() - timedelta(days=730)
start_date = (date).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')
interval = "1d"

data_records = {ticker: sort_fin_data(ticker, start_date, end_date, interval) for ticker in tickers}

with open('long_data_records.pkl', 'wb') as file:
    pickle.dump(data_records, file)
