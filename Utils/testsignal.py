import numpy as np
import pandas as pd

def test_signal(signals, y_valid, margin):
    # Calculate returns based on trading signals
    returns = pd.Series(index=y_valid.index)
    for i in range(1, len(signals)):
        if signals[i-1] > 0:  # Buy signal
            returns.iloc[i] = (y_valid.iloc[i] - y_valid.iloc[i-1]) / y_valid.iloc[i-1]
        else:  # Sell signal
            returns.iloc[i] = (y_valid.iloc[i-1] - y_valid.iloc[i]) / y_valid.iloc[i-1]

    # Calculate directional accuracy
    actual_changes = np.sign(np.diff(y_valid))
    predicted_changes = np.sign(np.diff(signals))
    correct_directions = np.sum(actual_changes == predicted_changes)
    directional_accuracy = correct_directions / len(actual_changes)
    directional_accuracy_percentage = directional_accuracy * 100

    # Calculate Sharpe ratio
    sharpe_ratio = returns.mean() / returns.std()
    
    # Calculate cumulative returns starting with $100
    cumulative_returns = (1 + returns).cumprod() * margin
    cumulative_returns = cumulative_returns[1:]  # since first day no signal
    
    # Calculate annualized returns
    annualized_returns = (cumulative_returns.iloc[-1] / cumulative_returns.iloc[0]) ** (12 / len(cumulative_returns)) - 1

    # Calculate maximum drawdown
    cumulative_max = cumulative_returns.cummax()
    drawdown = (cumulative_max - cumulative_returns) / cumulative_max
    max_drawdown = drawdown.max()

    return {
        'returns': cumulative_returns,
        'directional_accuracy': round(directional_accuracy_percentage, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'annualized_returns': round(annualized_returns * 100, 2),
        'max_drawdown': round(max_drawdown * 100, 2)
    }
