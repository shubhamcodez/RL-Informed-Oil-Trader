import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def calculate_volatility(returns, window=2):
    return returns.rolling(window=window).std()

def train_volatility_model(X_train, y_train, window=2):
    returns = y_train.pct_change().dropna()
    vol_target = calculate_volatility(returns, window).dropna()
    features = X_train.iloc[1:len(vol_target)+1].values  # Use lagged features
    vol_model = RandomForestRegressor()
    vol_model.fit(features, vol_target)
    return vol_model

def predict_volatility(vol_model, X_valid):
    features = X_valid.values
    return vol_model.predict(features)

def volatility_based_adjustment(predictions, predicted_volatility, vol_threshold=0.25):
    adjusted_predictions = predictions.copy()
    for i in range(len(predictions)):
        if predicted_volatility[i] > vol_threshold:
            adjusted_predictions[i] = 0
    return adjusted_predictions
