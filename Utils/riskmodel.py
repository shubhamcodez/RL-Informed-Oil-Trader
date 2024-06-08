import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def calculate_volatility(returns, window=20):
    return returns.rolling(window=window).std()

def train_volatility_model(X_train, y_train, window=20):
    returns = y_train.pct_change().dropna()
    vol_target = calculate_volatility(returns, window).dropna()
    features = X_train.iloc[1:len(vol_target)+1].values  # Use lagged features
    vol_model = RandomForestRegressor()
    vol_model.fit(features, vol_target)
    return vol_model

def predict_volatility(vol_model, X_valid):
    features = X_valid.values
    return vol_model.predict(features)

def volatility_based_adjustment(returns, predicted_volatility, vol_threshold):
    positions = pd.Series(index=returns.index, data=1)
    for i in range(1, len(returns)):
        if predicted_volatility[i] > vol_threshold:
            #positions[i] = adjusted_position()
            positions[i] = 0
            break
    adjusted_returns = returns * positions.shift(1).fillna(1)
    return adjusted_returns



