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
vol_prediction = riskModel.predict_volatility(vol_model, X_valid) #predict volitality for next day
risk_adjusted_prediction = riskModel.volatility_based_adjustment(prediction, vol_prediction) #if expected volitality is high, then close the position

# Calculate final metrics
final_results = ts.test_signal(risk_adjusted_prediction, y_valid, Margin)

# Plot cumulative returns with metrics
plt.figure(figsize=(12, 8))
plt.plot(final_results['returns'].index, final_results['returns'], label='Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.title('Performance of the Strategy')

# Display the metrics on the plot
metrics_text = (f"Directional Accuracy: {final_results['directional_accuracy']}%\n"
                f"Sharpe Ratio: {final_results['sharpe_ratio']:.2f}\n"
                f"Annualized Returns: {final_results['annualized_returns']:.2f}%\n"
                f"Max Drawdown: {final_results['max_drawdown']:.2f}%")

plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))

plt.show()
