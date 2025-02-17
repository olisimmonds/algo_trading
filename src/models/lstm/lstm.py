import numpy as np
from sklearn.neural_network import MLPRegressor

def get_train_test_for_lstm(scaled_prices):
    X_train, y_train = np.array(scaled_prices[:-2]).reshape(-1, 1), scaled_prices[1:-1]
    return X_train, y_train

def build_model(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_future(model, data, scaler):
    # Predict the next value
    prediction = model.predict(np.array(data[-1]).reshape(-1, 1))    
    prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
    return prediction