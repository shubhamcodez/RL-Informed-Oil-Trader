import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import gym

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

def train_rl_agent(returns, volatility):
    """
    Train a reinforcement learning agent to determine position sizes.
    
    Parameters:
    returns (pd.Series): Adjusted returns after applying risk management.
    volatility (np.ndarray): Predicted volatilities.
    
    Returns:
    PPO: Trained reinforcement learning agent.
    """
    env = DummyVecEnv([lambda: TradingEnv(returns, volatility)])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)
    return model

def apply_position_sizing(agent, returns, volatility, margin):
    """
    Apply the trained reinforcement learning agent for position sizing.
    
    Parameters:
    agent (PPO): Trained reinforcement learning agent.
    returns (pd.Series): Adjusted returns after applying risk management.
    volatility (np.ndarray): Predicted volatilities.
    margin (float): Available margin for trading.
    
    Returns:
    pd.Series: Position sizes determined by the RL agent.
    """
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

def positional_rl_bot(action, margin):
    """
    Determine adjustment percentage based on the action chosen by the RL agent and available margin.
    
    Parameters:
    action (int): Action chosen by the RL agent (0 for hold, 1 for buy, 2 for sell).
    margin (float): Available margin for trading.
    
    Returns:
    float: Adjustment percentage.
    """
    if action == 0:  # Hold
        return 0
    elif action == 1:  # Buy
        return min(1, margin / 10)  # Adjust up to 10% of the margin for buying
    else:  # Sell
        return min(1, margin / 10)  # Adjust up to 10% of the margin for selling
