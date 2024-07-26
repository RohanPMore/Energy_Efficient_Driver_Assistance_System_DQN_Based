import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Output is typically bounded (e.g., -1 to 1 for actions)

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the Actor-Critic agent
class ActorCriticAgent:
    def __init__(self, env, actor, critic, actor_lr=0.001, critic_lr=0.001, gamma=0.99):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_mean = self.actor(state)
        action_std = torch.tensor(0.1)  # Optional: Add exploration noise with a fixed std
        action = torch.normal(action_mean, action_std)
        return action.detach().numpy()

    def train(self, max_episodes=1000, max_steps=1000):
        for episode in range(max_episodes):
            state = self.env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                state_tensor = torch.tensor(state, dtype=torch.float32)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                reward_tensor = torch.tensor(reward, dtype=torch.float32)

                # Compute TD error and update Critic
                current_value = self.critic(state_tensor)
                next_value = self.critic(next_state_tensor) if not done else 0
                target_value = reward_tensor + self.gamma * next_value
                critic_loss = nn.MSELoss()(current_value, target_value.detach())
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Compute policy loss and update Actor
                advantage = target_value - current_value
                log_prob = torch.distributions.Normal(action_mean, action_std).log_prob(action)
                actor_loss = -(log_prob * advantage.detach()).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                state = next_state
                if done:
                    break

            print(f"Episode {episode}: Reward = {episode_reward}")

# Main execution
if __name__ == "__main__":
    # Create environment
    env = gym.make('CartPole-v1')  # Replace with your environment

    # Define input and output dimensions
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]

    # Initialize Actor and Critic networks
    actor = Actor(input_dim, output_dim)
    critic = Critic(input_dim)

    # Initialize Actor-Critic agent
    agent = ActorCriticAgent(env, actor, critic)

    # Train the agent
    agent.train()
