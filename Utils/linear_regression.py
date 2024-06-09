import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Dates'] = pd.to_datetime(df['Dates'])
    df.set_index('Dates', inplace=True)
    df.fillna(0, inplace=True)
    return df

def split_data(df, target_column, test_size=0.10):
    features = df.drop(columns=[target_column])
    target = df[target_column]
    X_train, X_valid, y_train, y_valid = train_test_split(features, target, test_size=test_size, shuffle=False)
    return X_train, X_valid, y_train, y_valid

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_valid, y_valid):
    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    mae = mean_absolute_error(y_valid, y_pred)
    return {'mse': mse, 'mae': mae}
