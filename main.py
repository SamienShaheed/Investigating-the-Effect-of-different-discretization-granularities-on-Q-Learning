import gym
from utils.dqn import DQN, train_dqn

def main():
    # Set the environment to MountainCar-v0
    env = gym.make('MountainCar-v0')

    # Set the hyperparameters for training
    episodes = 1000
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    learning_rate = 1e-3
    memory_size = 10000
    batch_size = 64
    target_update = 10

    # Determine the size of the state space and the number of actions
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Instantiate the DQN model
    policy_net = DQN(state_dim, n_actions)
    
    # Train the DQN model using the Mountain Car Environment
    trained_policy, embeddings = train_dqn(env, state_dim, n_actions, episodes, gamma,
                                           epsilon_start, epsilon_end, epsilon_decay,
                                           learning_rate, memory_size, batch_size, target_update)
    
    # After training, you can save the trained model and embeddings
    # torch.save(trained_policy.state_dict(), 'trained_policy.pth')
    # np.save('embeddings.npy', embeddings)

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
