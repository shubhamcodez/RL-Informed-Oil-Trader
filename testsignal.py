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
    #print(cumulative_returns)
    # Calculate annualized returns
    annualized_returns = (cumulative_returns.iloc[-1] / cumulative_returns.iloc[0]) ** (12 / len(cumulative_returns)) - 1

    # Calculate maximum drawdown
    cumulative_max = cumulative_returns.cummax()
    drawdown = (cumulative_max - cumulative_returns) / cumulative_max
    max_drawdown = drawdown.max()

    return {
        'directional_accuracy': round(directional_accuracy_percentage,2),
        'sharpe_ratio': round(sharpe_ratio,2),
        'annualized_returns': round(annualized_returns*100,2),
        'max_drawdown': round(max_drawdown*100,2)
    }


def trade(signal, price, margin, trades):
    # Initialize cumulative performance metrics if it's the first trade
    if trades == 0:
        cumulative_returns = margin
        cumulative_max = margin
        max_drawdown = 0
        correct_directions = 0
    else:
        cumulative_returns = margin
        cumulative_max = max(cumulative_max, cumulative_returns)
        drawdown = (cumulative_max - cumulative_returns) / cumulative_max
        max_drawdown = max(drawdown, max_drawdown)
        if signal * price > 0:
            correct_directions += 1

    # Calculate Sharpe ratio (not meaningful for single trades)
    sharpe_ratio = np.nan
    
    # Calculate directional accuracy
    if trades > 0:
        directional_accuracy = (correct_directions / trades) * 100
    else:
        directional_accuracy = 0

    # Update margin
    margin += (price - price) / price * margin

    # Update trades count
    trades += 1

    return {
        'directional_accuracy': round(directional_accuracy, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown': round(max_drawdown * 100, 2),
        'margin': round(margin, 2)
    }
