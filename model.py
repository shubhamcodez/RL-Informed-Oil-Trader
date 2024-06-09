import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Utils.linear_regression as alphaModel
import Utils.riskmodel as riskModel
import Utils.testsignal as ts

# Constants
Margin = 100000  # Amount of money available to trade

# Load and preprocess data
df = alphaModel.load_and_preprocess_data('Data/US_Crude_data_v1.csv')
X_train, X_valid, y_train, y_valid = alphaModel.split_data(df, 'Crude_Oil_Price')

# Train alpha model
alpha_model = alphaModel.train_model(X_train, y_train)
# Train risk model (predict volatility)
vol_model = riskModel.train_volatility_model(X_train, y_train)


prediction = alpha_model.predict(X_valid) #predict price
vol_prediction = riskModel.predict_volatility(vol_model, X_valid) #predict volitality 
risk_adjusted_prediction = riskModel.volatility_based_adjustment(prediction, vol_prediction) #if expected volitality is high, then close the position

# Calculate final metrics for the baseline and risk-adjusted strategy
res1 = ts.test_signal(prediction, y_valid, Margin) # Baseline
results = ts.test_signal(risk_adjusted_prediction, y_valid, Margin) # Risk-adjusted strategy

# Plot cumulative returns for both strategies
plt.figure(figsize=(12, 8))
plt.plot(res1['returns'].index, res1['returns'], label='Baseline Returns')
plt.plot(results['returns'].index, results['returns'], label='Risk-Adjusted Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.title('Performance of the Strategies')

# Display the metrics for both strategies on the plot
baseline_metrics_text = (f"Baseline:\n"
                        f"Directional Accuracy: {res1['directional_accuracy']}%\n"
                        f"Sharpe Ratio: {res1['sharpe_ratio']:.2f}\n"
                        f"Annualized Returns: {res1['annualized_returns']:.2f}%\n"
                        f"Max Drawdown: {res1['max_drawdown']:.2f}%")

risk_adjusted_metrics_text = (f"Risk-Adjusted:\n"
                              f"Directional Accuracy: {results['directional_accuracy']}%\n"
                              f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
                              f"Annualized Returns: {results['annualized_returns']:.2f}%\n"
                              f"Max Drawdown: {results['max_drawdown']:.2f}%")

plt.text(0.05, 0.85, baseline_metrics_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))

plt.text(0.65, 0.85, risk_adjusted_metrics_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))

plt.show()

