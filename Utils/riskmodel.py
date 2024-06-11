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

# Define the function for volatility-based adjustment
def volatility_based_adjustment(predictions, predicted_volatility, margin, vol_threshold=0.25):
    adjusted_predictions = predictions.copy()
    for i in range(len(predictions)):
        if predicted_volatility[i] > vol_threshold:
            # Apply position sizing using PositionRL bot
            adjusted_predictions[i] = apply_position_sizing(agent, predictions, predicted_volatility, margin)[i]
    return adjusted_predictions

# Define the RL trading environment
class TradingEnv(gym.Env):
    def __init__(self, returns, volatility):
        super(TradingEnv, self).__init__()
        self.returns = returns
        self.volatility = volatility
        self.current_step = 0
        
        # Define action and observation space
        # Actions: [hold, buy, sell]
        self.action_space = spaces.Discrete(3)
        
        # Observations: [return, volatility]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        return np.array([self.returns[self.current_step], self.volatility[self.current_step]])
    
    def step(self, action):
        self.current_step += 1
        
        if self.current_step >= len(self.returns) - 1:
            done = True
        else:
            done = False
        
        reward = 0
        if action == 1:  # Buy
            reward = self.returns[self.current_step]
        elif action == 2:  # Sell
            reward = -self.returns[self.current_step]
        
        obs = np.array([self.returns[self.current_step], self.volatility[self.current_step]])
        return obs, reward, done, {}
    
    def render(self, mode='human'):
        pass

# Define the function to train the RL agent
def train_rl_agent(returns, volatility):
    env = DummyVecEnv([lambda: TradingEnv(returns, volatility)])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)
    return model

# Define the function to apply position sizing using the trained RL agent
def apply_position_sizing(agent, returns, volatility, margin):
    env = TradingEnv(returns, volatility)
    obs = env.reset()
    position_sizes = []
    
    for _ in range(len(returns)):
        action, _ = agent.predict(obs)
        adjustment_percentage = positional_rl_bot(action, margin)  # Get adjustment percentage from PositionRL bot
        if action == 0:  # Hold
            position_size = 0
        elif action == 1:  # Buy
            position_size = adjustment_percentage
        else:  # Sell
            position_size = -adjustment_percentage
        position_sizes.append(position_size)
        obs, _, done, _ = env.step(action)
        if done:
            break    
    position_sizes = pd.Series(position_sizes, index=returns.index)
    return position_sizes

# Define the function for PositionRL bot
def positional_rl_bot(action, margin):
    if action == 0:  # Hold
        return 0
    elif action == 1:  # Buy
        return min(1, margin / 10)  # Adjust up to 10% of the margin for buying
    else:  # Sell
        return min(1, margin / 10)  # Adjust up to 10% of the margin for selling
