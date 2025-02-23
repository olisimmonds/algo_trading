import pandas as pd
import numpy as np
import mplfinance as mpf
from scipy.signal import argrelextrema
import pickle

def calculate_rsi(df, column='Close', period=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

import yfinance as yf
import pandas as pd

def get_previous_close(ticker, date):
    """
    Fetch the closing price of the previous trading day for a given stock ticker.
    """
    # Convert to datetime and subtract one day
    target_date = pd.to_datetime(date)
    start_date = target_date - pd.DateOffset(days=5)  # Fetching a range to ensure we get a valid previous trading day
    
    # Download historical data
    df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=target_date.strftime("%Y-%m-%d"))

    if df.empty:
        return None  # No data available

    # Get the last available close price before the given date
    previous_close = df['Close'].iloc[-1]  
    return previous_close


def detect_reversal_signals(df):
    """Identify reversal points based on RSI, downtrend, support, and doji patterns."""
    
    df['RSI'] = calculate_rsi(df)
    support_level = get_previous_close(df["Ticker"][0], df["Time"][0])

    reversal_points = []  # Stores indices where reversal signals occur

    for i in range(5, len(df) - 1):  # Start after at least 5 candlesticks
        # 1. Check if at least 5 candlesticks are red (closing lower than opening)
        if all(df['Close'][i-5:i] < df['Open'][i-5:i]):
            
            # 2. RSI must be below 10
            if df['RSI'][i] < 10:

                # 3. The stock must be trading near a support level
                # if abs((df['Close'][i] - support_level.values)/df['Close'][i]) < 0.01:
                    
                # 4. A green doji appears immediately after
                next_candle = df.iloc[i+1]
                body_size = abs(next_candle['Close'] - next_candle['Open'])
                wick_size = abs(next_candle['High'] - next_candle['Low'])
                
                if body_size < wick_size * 0.4 and next_candle['Close'] > next_candle['Open']:
                    reversal_points.append(i+1)  # Mark reversal point at next candle

    return reversal_points

def calculate_vwap(df):
    """Calculate VWAP (Volume Weighted Average Price)."""
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

def plot_reversal_signals(df):
    """Plot stock data and mark reversal points."""

    reversal_points = detect_reversal_signals(df)
    support = get_previous_close(df["Ticker"][0], df["Time"][0]).values

    # Calculate technical indicators
    df['9_EMA'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['20_EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['50_SMA'] = df['Close'].rolling(window=50).mean()
    df['200_SMA'] = df['Close'].rolling(window=200).mean()
    df['VWAP'] = calculate_vwap(df)

    reversal_points = detect_reversal_signals(df)
    support = get_previous_close(df["Ticker"][0], df["Time"][0]).values

    # Create markers for detected points
    marker_series = [np.nan] * len(df)
    for idx in reversal_points:
        marker_series[idx] = df['Low'][idx] * 0.995  # Place marker slightly below the low

    # Ensure datetime index
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    
    # Create an additional plot for RSI
    ap_rsi = mpf.make_addplot(df['RSI'], panel=1, color='blue', ylabel="RSI")
    
    # Create an additional plot for reversal markers
    ap_markers = mpf.make_addplot(marker_series, type='scatter', marker='^', color='red', markersize=100)
    support_marker = [float(support) for i in range(len(df))]
    ap_support = mpf.make_addplot(support_marker, type='scatter', marker='o', color='green', markersize=10)

    ap_9_ema = mpf.make_addplot(df['9_EMA'], color='orange')
    ap_20_ema = mpf.make_addplot(df['20_EMA'], color='purple')
    ap_50_sma = mpf.make_addplot(df['50_SMA'], color='blue')
    # ap_200_sma = mpf.make_addplot(df['200_SMA'], color='red')
    ap_vwap = mpf.make_addplot(df['VWAP'], color='green')

    # Plot everything
    mpf.plot(df, type="candle", volume=True, style="charles",
             title="Reversal Trading Strategy",
             ylabel="Price", ylabel_lower="Volume",
             addplot=[ap_rsi, ap_markers, ap_support, ap_9_ema, ap_20_ema, ap_50_sma, ap_vwap],
             alines=dict(alines=[(df.index[idx], df['Low'][idx]) for idx in reversal_points], colors="red", linewidths=1))

# Load the data
with open('data_records.pkl', 'rb') as file:
    data_records = pickle.load(file)

for date, tickers_data in data_records.items():  
    for ticker, time_series in tickers_data.items():
        if len(time_series)>30:
            if len(detect_reversal_signals(time_series))>1:
                plot_reversal_signals(time_series)
