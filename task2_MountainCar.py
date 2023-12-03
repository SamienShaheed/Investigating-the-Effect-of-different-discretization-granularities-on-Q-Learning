import os
import gym
import time
import numpy as np
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Parameters
EPISODES = 10000
UPDATE_EVERY = 100


def QLearning(env, learning, discount, epsilon, min_eps, episodes, granularity, discrete_actions, state_visits, action_distribution):
    # start counter
    total_time_start = time.perf_counter()

    # Determine size of discretized state space
    num_states = (discrete_actions[-1] - discrete_actions[0]) * np.array(
        [10 * granularity, 100 * granularity])
    num_states = np.round(num_states, 0).astype(int) + 1

    # Initialize Q table
    Q = np.random.uniform(low=-1, high=1,
                          size=(num_states[0], num_states[1],
                                env.action_space.n))

    # Initialize variables to track rewards and times
    reward_list = []
    avg_episode_reward_list = []
    time_list = []
    avg_episode_time_list = []

    # Initialize dictionaries to track state visits and action distribution
    state_visits = {}
    action_distribution = {action: 0 for action in range(env.action_space.n)}

    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps) / episodes

    # Run Q learning algorithm
    for i in range(episodes):
        tic = time.perf_counter()
        # Initialize parameters
        terminated = truncated = False
        tot_reward, reward = 0, 0
        state, _ = env.reset()

        # Discretize state
        state_adj = (state - discrete_actions[0]) * np.array([10 * granularity, 100 * granularity])
        state_adj = np.round(state_adj, 0).astype(int)

        while not terminated and not truncated:
           # Before taking an action, we track the current state's visit count
            state_key = tuple(state_adj)  # Convert the state to a tuple to use as a dictionary key
            track_state_visits(state_key, state_visits)  # Track state visits
           
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            action_distribution[action] += 1  # Track action distribution
            
            # Get next state and reward
            state2, reward, terminated, truncated, _ = env.step(action)
            # update the reward, considering the original reward (-1 for each timestep) + energy stored * 100 as a
            # scaling factor to keep make both variables equally important (both to 1 decimal)
            reward += energy_stored(state2) * 1000

            # Discretize state2
            state2_adj = (state2 - discrete_actions[0]) * np.array([10 * granularity, 100 * granularity])
            state2_adj = np.round(state2_adj, 0).astype(int)

            # Allow for terminal states
            if terminated:
                Q[state_adj[0], state_adj[1], action] = reward + 100
            # Adjust Q value for current state
            else:
                delta = learning * (reward +
                                    discount * np.max(Q[state2_adj[0], state2_adj[1]]) -
                                    Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1], action] += delta

            # Update variables
            tot_reward += reward
            state_adj = state2_adj

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction

        # Calculate exploration entropy after the episode ends
        entropy = calculate_entropy(action_distribution)
        
        # Track rewards and time
        toc = time.perf_counter()
        reward_list.append(tot_reward)
        time_list.append(toc - tic)

        if (i + 1) % UPDATE_EVERY == 0:
            avg_episode_reward = np.mean(reward_list)
            avg_episode_reward_list.append(avg_episode_reward)
            reward_list = []

            # Track time taken
            avg_episode_time = np.mean(time_list)
            avg_episode_time_list.append(avg_episode_time)
            time_list = []

            # After each episode
            entropy = calculate_entropy(action_distribution)
            print(f"Entropy after episode {i}: {entropy}")
            
            # print results + time every 100 episodes
            print(f"""Episode: {i + 1}
    Average Reward: {avg_episode_reward}
    Average Time Taken: {avg_episode_time:0.7f}s""")

    env.close()

    total_time_end = time.perf_counter()
    total_time = total_time_end - total_time_start

    return avg_episode_reward_list, avg_episode_time_list, total_time

def output(rewards, times, total_time, granularity='1x', run="1"):
    results = f"""Granularity: {granularity}
    Total time: {total_time:0.7f}s
    Best average reward: {rewards[-1]}
    Best time: {times[-1]}s
    Stability: {stability(rewards)}%"""

    print(results)

    here = os.path.dirname(os.path.realpath(__file__))
    run_directory = os.path.join(here, f"run_{run}")
    subdirectory = os.path.join(run_directory, granularity)
    filepath = os.path.join(subdirectory, f"{granularity}_experiment_results.txt")

    if not os.path.isdir(run_directory):
        os.mkdir(run_directory)
    if not os.path.isdir(subdirectory):
        os.mkdir(subdirectory)

    with open(filepath, 'w') as f:
        print(results, file=f)
        print("\n" + f"Rewards: {rewards}", file=f)
        print("\n" + f"Times: {times}", file=f)

    # Plot Rewards
    plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig(os.path.join(subdirectory, f"{granularity}_rewards.jpg"))
    plt.close()

    # Plot Time
    plt.plot(100 * (np.arange(len(rewards)) + 1), times, color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Average Time (in seconds)')
    plt.title('Average Time vs Episodes')
    plt.savefig(os.path.join(subdirectory, f"{granularity}_times.jpg"))
    plt.close()

    with open(os.path.join(here, f"all_experiment_results.txt"), 'a') as f:
        print(f"\n\nRun: {run}:\n\n", file=f)
        print(f"{results}", file=f)

    return results

def energy_stored(observations):
    # we use energy stored as a way to additionally reward the agent

    # set mass as 1 to keep code readable
    mass = 1
    # gravity pre-determined by the environment
    gravity = 0.0025
    position = observations[0]
    velocity = observations[1]
    kinetic = 0.5 * mass * (velocity ** 2)
    potential = mass * gravity * np.cos(3 * position)

    # as there is no friction in the system
    # total energy = kinetic energy + potential energy
    return kinetic + potential

def stability(rewards):
    # using relative error to measure the avg rate of change of rewards per 100 episodes,
    # to understand the stability of the algorithm
    # 1 - |(x2 - x1) - (x3 - x2)| / (x2 - x1))
    stability = 0
    for i in range(1, len(rewards) - 1):
        delta1 = abs(rewards[i] - rewards[i - 1]) / 100
        delta2 = abs(rewards[i + 1] - rewards[i]) / 100
        stability += abs(delta2 - delta1) / abs(delta2)

    # len(rewards) + 2 because we skipped the first and last rewards in the loop
    avg_stability = stability / (len(rewards) + 2)
    return avg_stability * 100

def run_experiment(state_granularity, action_granularity, run="1"):
    # Discretize the action space
    discrete_actions = discretize_action_space(env, action_granularity)
    
    # Initialize tracking for state visits and action distribution
    state_visits = {}
    action_distribution = {action: 0 for action in range(len(discrete_actions))}

    # Run the QLearning algorithm
    rewards, times, total_time = QLearning(env, 0.2, 0.9, 0.8, 0, EPISODES, action_granularity, discrete_actions, state_visits, action_distribution)
    
    # Calculate exploration metrics
    coverage = len(state_visits) / np.prod(env.observation_space.shape)
    entropy = calculate_entropy(action_distribution)

    # Output the results
    output(rewards, times, total_time, state_granularity, run, entropy, coverage)
    
    # Return metrics for further analysis
    return rewards, times, total_time, entropy, coverage

# Function to discretize action space of given Environment
def discretize_action_space(env, num_actions):
    if isinstance(env.action_space, gym.spaces.Box):
        # For a continuous action space, use the 'high' and 'low' attributes
        high = env.action_space.high[0]
        low = env.action_space.low[0]
        discrete_actions = np.linspace(low, high, num_actions)
    elif isinstance(env.action_space, gym.spaces.Discrete):
        # For a discrete action space, simply use the range of actions
        discrete_actions = np.arange(env.action_space.n)
    else:
        raise NotImplementedError("Action space type not supported for discretization")
    return discrete_actions


def track_state_visits(state, state_visits):
    state = tuple(state)
    state_visits[state] = state_visits.get(state, 0) + 1

def calculate_entropy(action_distribution):
    probabilities = np.array(list(action_distribution.values())) / sum(action_distribution.values())
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy

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
        plt.plot(100 * (np.arange(EPISODES / UPDATE_EVERY) + 1),
             outcomes[granularity][0], label=f"{granularity}")
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig(os.path.join(here, f"rewards.jpg"))
    plt.close()

    for granularity in granularities:
        plt.plot(100 * (np.arange(EPISODES / UPDATE_EVERY) + 1),
             outcomes[granularity][1], label=f"{granularity}")
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Time (in seconds)')
    plt.title('Average Time vs Episodes')
    plt.savefig(os.path.join(here, f"times.jpg"))
    plt.close()
