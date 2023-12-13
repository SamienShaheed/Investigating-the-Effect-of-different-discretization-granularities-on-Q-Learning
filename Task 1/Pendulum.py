import gym
import numpy as np
import matplotlib.pyplot as plt

# Create the Pendulum environment
env = gym.make('Pendulum-v1', render_mode='human')
env.reset()
episodes = 200

# Initialize the list for accumulated rewards
episode_rewards = []

for i in range(episodes):
    env.render()
    total_reward = 0  # Initialize the total reward for this episode
    
    # Assume the episode lasts for a fixed number of time steps
    time_steps = 200
    for _ in range(time_steps):
        # Sample a random action
        action = env.action_space.sample()
        # Take the action and observe the new state and reward
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward  # Accumulate the reward
        
        if done:
            break  # If the environment says the episode is done, exit the loop

    episode_rewards.append(total_reward)  # Append the total reward for this episode
    print(f"Reward at Episode {i}: {total_reward}")
env.close()

# Plot Rewards
plt.plot(np.arange(len(episode_rewards)), episode_rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.savefig('outcomes_Pendulum.jpg')
plt.close()
