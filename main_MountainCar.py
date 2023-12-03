import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from utils.plotter import output
from utils.qLearning import QLearning
from utils.entropy import calculate_entropy
from utils.discretize import discretize_action_space

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Parameters
episodes = 10000
logFrequency = 100 # Update results every 100 episodes
learningRate = 0.2
discountFactor = 0.9
epsilon = 0.8 # Adjust to control level of exploration
minEps = 0

def run_experiment(state_granularity, action_granularity, run="1"):
    # Discretize the action space
    discrete_actions = discretize_action_space(env, action_granularity)
    
    # Initialize tracking for state visits and action distribution
    state_visits = {}
    action_distribution = {action: 0 for action in range(len(discrete_actions))}

    # Run the QLearning algorithm
    rewards, times, total_time = QLearning(env, 
                                           learningRate, 
                                           discountFactor, 
                                           epsilon, 
                                           minEps, 
                                           episodes, 
                                           action_granularity, 
                                           discrete_actions, 
                                           state_visits, 
                                           action_distribution, 
                                           logFrequency)
    
    # Calculate exploration metrics
    coverage = len(state_visits) / np.prod(env.observation_space.shape)
    entropy = calculate_entropy(action_distribution)

    # Output the results
    output(rewards, times, total_time, state_granularity, run, entropy, coverage)
    
    # Return metrics for further analysis
    return rewards, times, total_time, entropy, coverage

# set up all granularities here so we can loop over them, instead of repeating code
granularities = {"Quarter": 0.25, "Half": 0.5, "1x": 1, "2x": 2, "5x": 5, "10x": 10, "100x": 100}
here = os.path.dirname(os.path.realpath(__file__))

for run in range(1, 4):
    # Run Q-learning algorithm at different granularities
    outcomes = {}

    # Loop over granularities
    for granularity in granularities:
        outcomes[granularity] = list(run_experiment(granularity, granularities[granularity], run))

    for granularity in granularities:
        plt.plot(100 * (np.arange(episodes / logFrequency) + 1),
             outcomes[granularity][0], label=f"{granularity}")