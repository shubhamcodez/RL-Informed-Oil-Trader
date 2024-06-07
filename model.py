import linear_regression as am
import testsignal as ts
import riskmodel 
import numpy as np
#import positionRL as ps
import pandas as pd
import matplotlib.pyplot as plt


Margin = 100000 #Amount of money available to trade

# Load and preprocess data
df = am.load_and_preprocess_data('US_Crude_data_v1.csv')
X_train, X_valid, y_train, y_valid = am.split_data(df, 'Crude_Oil_Price')
# Train alpha model
alpha_model = am.train_model(X_train, y_train)

#trading signal
predictions = alpha_model.predict(X_valid)
#print(ts.test_signal(predictions, y_valid, Margin))

# Train risk model (predict volatility)
#vol_model = rm.train_volatility_model(X_train, y_train)


'''
# Apply risk management
adjusted_returns = rm.apply_risk_management(y_valid, vol_model, X_valid, vol_threshold=0.02)

# Train reinforcement learning agent for position sizing
agent = ps.train_reinforcement_learning_agent(adjusted_returns, rm.predict_volatility(vol_model, X_valid))

# Apply position sizing
position_sizes = ps.apply_position_sizing(agent, adjusted_returns, rm.predict_volatility(vol_model, X_valid))

# Calculate final returns based on position sizes
final_returns = adjusted_returns * position_sizes

# Calculate cumulative returns
cumulative_returns = (1 + final_returns).cumprod() * 100

# Plot cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Returns with Risk Management and Position Sizing')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.title('Cumulative Returns with Complete Quantitative Model')
plt.show()
'''