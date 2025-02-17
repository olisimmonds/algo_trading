import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.lstm.lstm import get_train_test_for_lstm, build_model, predict_future

def predict_stock(stock_vals):
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(stock_vals)

    X_train, y_train = get_train_test_for_lstm(scaled_prices)
    
    # Build the MLPRegressor model
    model = build_model(X_train, y_train.ravel())
    
    predictions = predict_future(model, y_train, scaler) # change y_tain to whole data set when real data

    # Record key values
    last_train_price = scaler.inverse_transform(np.array(X_train[-1][0]).reshape(-1, 1))[0][0]
    last_predicted_price = predictions[-1][0]
    last_actual_price = scaler.inverse_transform(np.array(scaled_prices[-1][0]).reshape(-1, 1))[0][0]
    return last_train_price, last_predicted_price, last_actual_price