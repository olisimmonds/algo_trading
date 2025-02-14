import pandas as pd
import re
import yfinance as yf

def get_s_p_tickers():
    # URL of the Wikipedia page
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    # Read the tables on the page
    tables = pd.read_html(url)

    # The first table contains the list of S&P 500 companies
    df = tables[0]

    # Extract the 'Symbol' column
    symbols = df['Symbol'].tolist()

    # Filter out symbols containing any special character
    symbols = [symbol for symbol in symbols if re.match(r'^[A-Za-z]+$', symbol)]

    return symbols

def get_fin_data(ticker, start, end, interval, clossing=False):
    data = yf.download(
        tickers=ticker, 
        start=start, 
        end=end, 
        interval=interval
    )

    if clossing:
        # Extract and clean the closing prices
        closing_prices = data["Close"].dropna().values.reshape(-1, 1)

        return closing_prices
    return data

