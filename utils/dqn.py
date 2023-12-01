import numpy as np
from collections import deque
from collections import namedtuple
import random

import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture of DQN Model
class DQN(nn.Module):
    def __init__(self, space_dim, n_actions, out_feature=24):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(space_dim, out_feature)
        self.fc2 = nn.Linear(out_feature, out_feature)
        self.fc3 = nn.Linear(out_feature, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

Transition = namedtuple('Transition',('state', 'action', 'reward', 'next_state', 'done'))

# Function to train the dqn to given environment
def train_dqn(env, state_dim, action_dim, episodes, gamma, epsilon_start, epsilon_end, epsilon_decay, learning_rate, memory_size, batch_size, target_update):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize policy and target networks
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target net is not trained
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = deque(maxlen=memory_size)  # Experience replay memory
    epsilon = epsilon_start
    
    # Define function to select action using epsilon-greedy policy
    def select_action(state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor([state], device=device, dtype=torch.float)
                q_values = policy_net(state)
                return q_values.max(1)[1].view(1, 1).item()
        else:
            return env.action_space.sample()
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        embeddings = []  # To store embeddings
        total_reward = 0
        
        while True:
            action = select_action(state, epsilon)
            result = env.step(action)
            print(result)
            next_state, reward, done, done2, _ = env.step(action)
            total_reward += reward
            
            # Store the transition in memory
            memory.append((state, action, reward, next_state, done))
            
            # Move to the next state
            state = next_state
            
            # Perform one step of the optimization
            if len(memory) > batch_size:
                transitions = random.sample(memory, batch_size)
                batch = Transition(*zip(*transitions))
                
                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                
                state_action_values = policy_net(state_batch).gather(1, action_batch)
                
                # Compute V(s_{t+1}) for all next states.
                next_state_values = torch.zeros(batch_size, device=device)
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
                
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * gamma) + reward_batch
                
                # Compute Huber loss
                loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update embeddings for the last hidden layer
            with torch.no_grad():
                state_t = torch.tensor([state], device=device, dtype=torch.float)
                embedding = policy_net.fc[0:4](state_t)  # Assume second last layer output is the embedding
                embeddings.append(embedding.cpu().numpy())
            
            if done:
                break
        
        # Update epsilon
        epsilon = max(epsilon_end, epsilon_decay*epsilon)
        
        # Update the target network, copying all weights and biases in DQN
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Record episode stats (reward, embeddings, etc.)
        # ...
        
    return policy_net, embeddings