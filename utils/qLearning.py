import time
import numpy as np

from utils.entropy import calculate_entropy

# Track the number of times each state is visited
def track_state_visits(state, state_visits):
    state = tuple(state)
    state_visits[state] = state_visits.get(state, 0) + 1

# Energy stored is used as an additional reward for MountainCar environment
def energy_stored(observations):    
    mass = 1 # Set mass to 1 for convenience
     
    # Gravity pre-determined in environment
    gravity = 0.0025 # Given value in MountainCar Environment
    position = observations[0]
    velocity = observations[1]
    kinetic = 0.5 * mass * (velocity ** 2)
    potential = mass * gravity * np.cos(3 * position)

    # Total Energy = Kinetic energy + Potential energy
    return kinetic + potential

# Main Q-Learning Algorithm
def QLearning(env, learning, discount, epsilon, min_eps, episodes, granularity, discrete_actions, state_visits, action_distribution, logFrequency):
    # start counter
    total_time_start = time.perf_counter()

    # Determine size of discretized state space
    num_states = (discrete_actions[-1] - discrete_actions[0]) * np.array(
        [10 * granularity, 100 * granularity])
    num_states = np.round(num_states, 0).astype(int) + 1
    
    # Initialize Q table
    Q = np.random.uniform(low=-1, high=1,
                          size=(num_states[0], num_states[1],
                                len(discrete_actions)))

    # Initialize variables to track evaluation metrics
    reward_list = []
    avg_episode_reward_list = []
    
    time_list = []
    avg_episode_time_list = []
    
    entropy_list = []
    avg_episode_entropy_list = []
    
    coverage_list = []
    avg_episode_coverage_list = []

    ######################################################
    # Calculate episodic reduction in epsilon
    #reduction = (epsilon - min_eps) / episodes
    ######################################################

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
                action = np.random.randint(0, len(discrete_actions))
            
            #action_distribution = {action: 0 for action in range(len(discrete_actions))}
            action_distribution[action] += 1  # Track action distribution
            
            # Get next state and reward
            state2, reward, terminated, truncated, _ = env.step(np.array([action]))
            # update the reward, considering the original reward (-1 for each timestep) + energy stored * 100 as a
            # scaling factor to keep make both variables equally important (both to 1 decimal)
            reward += energy_stored(state2) * 100

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

        ###############################
        # Decay epsilon
        #if epsilon > min_eps:
        #   epsilon -= reduction
        ###############################
        
        # Calculate exploration entropy after the episode ends
        entropy = calculate_entropy(action_distribution)
        entropy_list.append(entropy) # Add entropy to list
        
        # Track rewards and time
        toc = time.perf_counter()
        reward_list.append(tot_reward)
        time_list.append(toc - tic)

        # Calculate average over every logFrequency (default 100)
        if (i + 1) % logFrequency == 0:
            avg_episode_reward = np.mean(reward_list)
            avg_episode_reward_list.append(avg_episode_reward)
            reward_list = []

            # Track time taken
            avg_episode_time = np.mean(time_list)
            avg_episode_time_list.append(avg_episode_time)
            time_list = []

            # Calculate Average Entropy
            avg_episode_entropy = np.mean(entropy_list)
            avg_episode_entropy_list.append(avg_episode_entropy)
            
            # print metrics every 100 episodes
            print(f"""Episode: {i + 1}
    Entropy after episode {i+1}: {avg_episode_entropy}
    Average Reward: {avg_episode_reward}
    Average Time Taken: {avg_episode_time:0.7f}s""")

    env.close()

    total_time_end = time.perf_counter()
    total_time = total_time_end - total_time_start

    return avg_episode_reward_list, avg_episode_time_list, total_time, avg_episode_entropy_list