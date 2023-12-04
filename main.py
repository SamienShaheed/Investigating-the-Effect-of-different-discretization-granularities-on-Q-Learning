import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from utils.plotter import output
from utils.qLearning import QLearning
from utils.discretize import discretize_action_space

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Parameters
minEps = 0 # Used for Epsilon Decay Calculation (Not Required in Experiment)
epsilon = 0.2 # Adjust to control level of exploration
episodes = 10000
logFrequency = 100 # Average results over every 100 episodes
learningRate = 0.2
discountFactor = 0.9
noOfRuns = 3 # Number of times to run the experiment

def run_experiment(granularity_label, granularity_value, run="1"):
    # Discretize the action space
    discrete_actions = discretize_action_space(env, granularity_value)
    
    # Initialize tracking for state visits and action distribution
    state_visits = {}
    action_distribution = {action: 0 for action in range(len(discrete_actions))}

    # Run the QLearning algorithm
    rewards, times, total_time, coverages, entropy = QLearning(env,
                                                               learningRate,
                                                               discountFactor,
                                                               epsilon,
                                                               minEps,
                                                               episodes,
                                                               granularity_value,
                                                               discrete_actions,
                                                               state_visits,
                                                               action_distribution,
                                                               logFrequency)
    

    # Output the results
    output(rewards, times, total_time, epsilon, entropy, coverages, granularity_label, run)
    
    # Return metrics for further analysis
    return rewards, times, total_time, entropy, coverages

# set up all granularities here so we can loop over them, instead of repeating code
granularities = {"Quarter": 0.25, "Half": 0.5, "1x": 1, "2x": 2, "5x": 5, "10x": 10, "100x": 100}
here = os.path.dirname(os.path.realpath(__file__))

for run in range(1, noOfRuns+1):
    # Run Q-learning algorithm at different granularities
    outcomes = {}
    
    # Loop over granularities
    for granularity in granularities:
        print(granularity)
        outcomes[granularity] = list(run_experiment(granularity, granularities[granularity], run))
    
    ######################################### Graph Plotting ######################################### 
    # Plot Graph of Reward over Episode for all granularities 
    for granularity in granularities:
        plt.plot(100 * (np.arange(episodes / logFrequency) + 1),
             outcomes[granularity][0], label=f"{granularity}")
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Rewards')
    plt.title('Average Reward vs Episodes')
    plt.savefig(os.path.join(here, f"rewards.jpg"))
    plt.close()

    # Plot Graph of Average Time over Episode for all granularities
    for granularity in granularities:
        plt.plot(100 * (np.arange(episodes / logFrequency) + 1),
             outcomes[granularity][1], label=f"{granularity}")
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Time (in seconds)')
    plt.title('Average Time vs Episodes')
    plt.savefig(os.path.join(here, f"times.jpg"))
    plt.close()

    # Plot Graph of Entropy over Episode for all granularities
    for granularity in granularities:
        plt.plot(100 * (np.arange(episodes / logFrequency) + 1),
             outcomes[granularity][3], label=f"{granularity}")
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Entropy')
    plt.title('Average Entropy vs Episodes')
    plt.savefig(os.path.join(here, f"entropy.jpg"))
    plt.close()

    # Plot Graph of Coverage over Episode for all granularities
    for granularity in granularities:
        plt.plot(100 * (np.arange(episodes / logFrequency) + 1),
             outcomes[granularity][4], label=f"{granularity}")
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Coverage')
    plt.title('Average Coverage vs Episodes')
    plt.savefig(os.path.join(here, f"coverage.jpg"))
    plt.close()
    
    