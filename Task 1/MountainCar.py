import gym
import numpy as np
import matplotlib.pyplot as plt

# Create the MountainCar environment
env = gym.make('MountainCar-v0')
env.reset()

# Define the number of episodes
episodes = 200
# To store the total reward per episode
episode_rewards = []

for i in range(episodes):
    total_reward = 0
    state, _ = env.reset()
    
    while True:
        env.render()
        
        # Basic strategy: if velocity is positive, push right; else, push left
        action = 2 if state[1] > 0 else 0

        # Take the action and observe the new state and reward
        next_state, reward, done, truncated, info = env.step(action)
        
        energy_reward = -next_state[0] + next_state[1]**2  # Potential + Kinetic energy

        # Add the energy-based reward to the standard reward
        total_reward += reward + energy_reward

        state = next_state

        if done:
            episode_rewards.append(total_reward)
            break

env.close()
    
# Plot Rewards
plt.plot(np.arange(episodes), episode_rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Total Reward per Episode')
plt.savefig('outcomes_MountainCar.jpg')
plt.close()