# Quantitative Trading System

## Overview
This project aims to develop a comprehensive quantitative trading system utilizing mathematical and statistical models to make systematic trading decisions. The system comprises three key components: Alpha Model, Risk Model, and Position Sizing. 
<br>
Each component plays a vital role in analyzing market data, predicting price movements, managing risk, and optimizing position sizes.

## Components of the trading system

### 1. Alpha Model 
The Alpha Model focuses on generating signals to predict future price movements. In this project, we utilize linear regression to forecast crude oil prices. We then use these forecasts as directional signals to make trading decisions. 
<br>
Linear regression was chosen because it performed the best in terms of performance, accuracy and Sharpe ratio compared to XGB, and deep learning architectures like LSTM, and Transformers. 

### 2. Risk Model
The Risk Model is essential for managing risk and protecting against potential losses. We employ machine learning techniques, such as Random Forest, to forecast market volatility. By analyzing historical data and market indicators, the model estimates future volatility levels, enabling us to assess and mitigate risk effectively.

### 3. Position Sizing (Reinforcement Learning)
Position Sizing plays a crucial role in determining the optimal size of trading positions based on expected returns and risk. We implement a reinforcement learning agent to dynamically adjust position sizes in response to changing market conditions. By optimizing position sizes, we aim to maximize returns while minimizing potential losses.

## Usage
To utilize this quantitative trading system:
1. Train the Alpha Model using historical price data.
2. Develop the Risk Model to forecast market volatility.
3. Implement the Position Sizing strategy using reinforcement learning.
4. Integrate the three components into a cohesive framework for automated trading.

## Contributing
Contributions to enhance and improve the quantitative trading system are welcome. Please feel free to submit pull requests or open issues to discuss potential improvements or new features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Disclaimer:** This project is for educational and research purposes only. Trading in financial markets involves significant risk and may not be suitable for all investors. The authors of this project are not responsible for any financial losses incurred as a result of using this system.