import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import pickle
from datetime import datetime, timedelta

def calculate_sma(df, window, column='Close'):
    """Calculate Simple Moving Average (SMA)."""
    return df[column].rolling(window=window).mean()

def detect_trading_signals(df):
    """Identify entry and exit points based on SMA crossovers."""
    
    df['50_SMA'] = calculate_sma(df, 50)
    df['200_SMA'] = calculate_sma(df, 200)

    entry_points = []
    exit_points = []
    
    position = False  # Track if we are in a trade

    for i in range(1, len(df)):
        # Entry: 50_SMA crosses above 200_SMA (Golden Cross)
        if df['50_SMA'].iloc[i-1] < df['200_SMA'].iloc[i-1] and df['50_SMA'].iloc[i] > df['200_SMA'].iloc[i]:
            entry_points.append((df.index[i], df['Close'].iloc[i]))
            position = True  # In a trade
            
        # Exit: 50_SMA crosses below 200_SMA (Death Cross)
        elif position and df['50_SMA'].iloc[i-1] > df['200_SMA'].iloc[i-1] and df['50_SMA'].iloc[i] < df['200_SMA'].iloc[i]:
            exit_points.append((df.index[i], df['Close'].iloc[i]))
            position = False  # Exit trade
    
    if len(exit_points) < len(entry_points):
        entry_points = entry_points[:len(exit_points)] 
        
    return entry_points, exit_points

def plot_trading_signals(df, ticker, entry_points, exit_points):
    """Plot stock data with entry and exit signals."""
    
    # Ensure datetime index
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)

    df['50_SMA'] = calculate_sma(df, 50)
    df['200_SMA'] = calculate_sma(df, 200)

    # Convert entry and exit points to marker lists
    entry_markers = [np.nan] * len(df)
    exit_markers = [np.nan] * len(df)

    for date, price in entry_points:
        entry_markers[date] = price * 0.99  # Slight offset for visibility

    for date, price in exit_points:
        exit_markers[date] = price * 0.99  

    # Additional plots for indicators
    ap_50_sma = mpf.make_addplot(df['50_SMA'], color='blue')
    ap_200_sma = mpf.make_addplot(df['200_SMA'], color='red')
    ap_entries = mpf.make_addplot(entry_markers, type='scatter', marker='^', color='green', markersize=10)
    ap_exits = mpf.make_addplot(exit_markers, type='scatter', marker='v', color='black', markersize=10)

    # Plotting
    mpf.plot(df, type="candle", volume=True, style="charles",
             title=f"Swing Trading Strategy - {ticker}",
             ylabel="Price", ylabel_lower="Volume",
             addplot=[ap_50_sma, ap_200_sma, ap_entries, ap_exits])

# Load the data
with open('long_data_records.pkl', 'rb') as file:
    data_records = pickle.load(file)

trade_results = []

for ticker, df in data_records.items():
    if len(df) >= 200:  # Ensure enough data for 200-day SMA calculation
        entry_points, exit_points = detect_trading_signals(df)
        # if len(entry_points)>0:
            
        #     plot_trading_signals(df, ticker, entry_points, exit_points)

        # Save trade results
        for (entry_date, entry_price), (exit_date, exit_price) in zip(entry_points, exit_points):
            trade_results.append((entry_date, entry_price, exit_date, exit_price))

# Print trade results for analysis
for trade in trade_results:
    print(f'Entry: {trade[0]} at {trade[1]:.2f}, Exit: {trade[2]} at {trade[3]:.2f}')

# Calculate percentage gains
percentage_gains = [(exit_price - entry_price) / entry_price * 100 for _, entry_price, _, exit_price in trade_results]

# Compute average percentage gain
average_gain = sum(percentage_gains) / len(percentage_gains) if percentage_gains else 0

print(f'Average Percentage Gain: {average_gain:.2f}%')
