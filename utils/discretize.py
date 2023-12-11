import gym
import numpy as np

# Base number of discrete action spaces
baseBins = 10

# Function to discretize action space of given Environment
def discretize_action_space(env, granularity):
    if isinstance(env.action_space, gym.spaces.Box):
        # For a continuous action space, use the 'high' and 'low' attributes
        high = env.action_space.high[0]
        low = env.action_space.low[0]
        
        # Calculate how many discrete action spaces for each granularity
        num_actions = int(baseBins * granularity)
        
        # Discretize the action space
        discrete_actions = np.linspace(low, high, num_actions)
    elif isinstance(env.action_space, gym.spaces.Discrete):
        # For a discrete action space, simply use the range of actions
        discrete_actions = np.arange(env.action_space.n)
    else:
        raise NotImplementedError("Action space type not supported for discretization")
    return discrete_actions