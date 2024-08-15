import random
from collections import namedtuple
import optuna
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from functions.rl.ev_env import EVEnv
from functions.rl.model import QNetwork, ReplayBuffer, Transition

# Define the objective function to minimize
def objective(trial, env, agent, track_data):
    # Define hyperparameters to be optimized
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    capacity = trial.suggest_categorical('capacity', [5000, 10000, 20000])

    #input_dim = env.observation_space
    #output_dim = env.action_space
    input_dim = env.observation_space.shape[0] ##new added 09.08
    output_dim = env.action_space.shape[0]

    # Initialize agent with the suggested hyperparameters
    agent = EVAgent(env, input_dim, output_dim, lr=lr, gamma=gamma, batch_size=batch_size, capacity=capacity)
    
    # Train the agent and return the negative total rewards (as Optuna tries to minimize)
    total_rewards = train_agent(env, agent, track_data, episodes=5000)
    return -sum(total_rewards) if total_rewards else float('inf')  # Return a large value if total_rewards is None or empty

# DQN Agent class
class EVAgent:
    def __init__(self, env, input_dim, output_dim, lr=0.001, gamma=0.99, batch_size=64, capacity=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.q_network = QNetwork(input_dim, output_dim).to(self.device)
        self.target_network = QNetwork(input_dim, output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()

        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(capacity)
        self.env = env
        #print("Nikita::::::::::::::::")

    def select_action(self, state, epsilon):
        legal_actions = self.env.get_legal_actions()
        if random.random() < epsilon:            
            return random.choice(legal_actions)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            #q_values_masked = [q_values[0][a] if a in legal_actions else torch.tensor(float('-inf')) for a in range(self.input_dim)]
            q_values_masked = [q_values[0][a] if a in legal_actions else torch.tensor(float('-inf')).to(self.device) for a in range(self.output_dim)]
            return max(enumerate(q_values_masked), key=lambda x: x[1])[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.next_state if s is not None]).to(self.device)
        
        state_batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.state]).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)

        #state_action_values = self.q_network(state_batch).gather(1, action_batch) ##change due to error 10.08
        state_action_values = self.q_network(state_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()

        # Ensure both tensors have the same shape for correct loss calculation
        expected_state_action_values = torch.zeros((self.batch_size, 1), device=self.device)
        expected_state_action_values[non_final_mask, :] = (next_state_values.view(-1, 1) * self.gamma) + reward_batch

        # Calculate MSE loss
        loss = self.loss_function(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        pass

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.push(state, action, next_state, reward, done)

# Training loop
def train_agent(env, agent, track_data, episodes=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update=100): #episodes 1000
    epsilon = epsilon_start
    total_rewards = []
    max_steps = len(track_data['v_max'])  # R Set max_steps based on track_data['v_max']

    final_episode_states = {} # R Initialize an empty dictionary to store next_state values during the final episode

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        if episode == episodes - 1: # Track next_state values for the final episode
            store_next_state = True
        else:
            store_next_state = False

        for step in range(max_steps):
            #env.render()  
            #time.sleep(1)
            action = agent.select_action(state, epsilon)
            print(f"Selected action: {action}")
            #next_state, reward, done, _ = env.step(action)       temporary modification 09.08.2024         
            next_state, reward, done = env.step(action)   
            print(f"New state: {next_state}, Velocity: {next_state[0]}, Reward: {reward}")   
            total_reward += reward
            if store_next_state: # Store next_state in the dictionary for the final episode
                final_episode_states[step] = next_state
            agent.store_transition(state, action, next_state, reward, done)
            agent.update()
            state = next_state

            if done:
                break

        total_rewards.append(total_reward)
        #final_score = env.scoreboard_model.scoreboard_data['Full Score']['user_score'] 
        final_score = total_rewards
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % target_update == 0:
            agent.update_target_network()

        #if episode % 500 == 0:
            #max_steps += 50
        print(f"Episode {episode}/{episodes}, Total Reward: {total_reward}, Final Score: {final_score}")

    return total_rewards, final_episode_states

if __name__ == "__main__":
    # Create the EV environment and agent
    env = EVEnv()
    input_dim = env.observation_space
    output_dim = env.action_space
    agent = EVAgent(env, input_dim, output_dim)

    # Create an Optuna study object and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, env, agent), n_trials=10)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # Train the agent using the best hyperparameters found by Optuna
    best_agent = EVAgent(input_dim, output_dim, lr=best_params['lr'], gamma=best_params['gamma'],
                            batch_size=best_params['batch_size'], capacity=best_params['capacity'])
    total_rewards = train_agent(env, best_agent, episodes=2) #episode = 10000