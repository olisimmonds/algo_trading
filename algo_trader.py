import pandas as pd
from datetime import datetime, timedelta
from src.core_code.get_fin_info import get_s_p_tickers
from src.core_code.testing import calc_difference, plot_skatter, breached_threshold
from src.core_code.prediction import predict_stock

# Ticker details
tickers = get_s_p_tickers()

ave_act_dif = []
profits_005 = []
profits_01 = []
profits_015 = []
for i in range(30):
    results = []
    act_dif = []
    interval = "5m"
    start = (datetime.today() - timedelta(days=i+1)).strftime('%Y-%m-%d')
    end = (datetime.today() - timedelta(days=i)).strftime('%Y-%m-%d')

    try:
        # Generate plots and record results for each ticker
        for ticker in tickers:
            last_train_price, last_predicted_price, last_actual_price = predict_stock(ticker, start, end, interval)

            predicted_difference = calc_difference(last_train_price, last_predicted_price)
            actual_difference = calc_difference(last_train_price, last_actual_price)
            act_dif.append(actual_difference)

            breached_threshold(0.005, profits_005, predicted_difference, i, last_train_price, last_predicted_price, last_actual_price)
            breached_threshold(0.01, profits_01, predicted_difference, i, last_train_price, last_predicted_price, last_actual_price)
            breached_threshold(0.015, profits_015, predicted_difference, i, last_train_price, last_predicted_price, last_actual_price)

            results.append({
                "Ticker": ticker,
                "Predicted Difference": predicted_difference,
                "Actual Difference": actual_difference
            })

        df = pd.DataFrame(results)
        plot_skatter(df)

    except Exception as e:
        print(e)
    if len(act_dif)>0:
        ave_act_dif.append((i, sum(act_dif)/len(act_dif)))