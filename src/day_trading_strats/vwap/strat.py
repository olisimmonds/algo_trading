import pandas as pd
import numpy as np
import mplfinance as mpf
import pickle
import yfinance as yf

def calculate_vwap(df):
    """Calculate VWAP (Volume Weighted Average Price)."""
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

def detect_vwap_support(df, threshold=0.005, lookback_period=12):
    """
    Determine if the VWAP is acting as support.
    VWAP is considered support when the price stays close to the VWAP for `lookback_period` candles
    and then closes above the VWAP.
    """
    df['VWAP'] = calculate_vwap(df)

    support_points = []  # List to store support points (index positions)

    for i in range(lookback_period, len(df)):  # Start from `lookback_period` because we need history
        if all(abs(df['Close'][i - j] - df['VWAP'][i - j]) / df['VWAP'][i - j] < threshold for j in range(lookback_period)):
            if all(df['Close'][i-j] > df['VWAP'][i-j] for j in range(lookback_period)):
                support_points.append(i)  # Mark this index as a support point

    return support_points

def plot_vwap_support(df):
    """Plot stock data and mark VWAP support points."""
    support_points = detect_vwap_support(df)

    # Create markers for detected VWAP support points
    marker_series = [np.nan] * len(df)
    for idx in support_points:
        marker_series[idx] = df['Low'][idx] * 0.995  # Place marker slightly below the low

    # Ensure datetime index
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)

    # Create an additional plot for VWAP
    ap_vwap = mpf.make_addplot(df['VWAP'], color='green')
    
    # Create an additional plot for VWAP support markers
    ap_support = mpf.make_addplot(marker_series, type='scatter', marker='^', color='red', markersize=100)

    # Plot everything
    mpf.plot(df, type="candle", volume=True, style="charles",
             title="VWAP Support Strategy",
             ylabel="Price", ylabel_lower="Volume",
             addplot=[ap_vwap, ap_support])

# Load the data
with open('data_records.pkl', 'rb') as file:
    data_records = pickle.load(file)

for date, tickers_data in data_records.items():  
    for ticker, time_series in tickers_data.items():
        if len(time_series) > 30:  # Ensure enough data is present
            if len(detect_vwap_support(time_series))>0:
                plot_vwap_support(time_series)
