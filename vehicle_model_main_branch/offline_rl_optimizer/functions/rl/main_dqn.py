# Import the smooth_curve_function from the separate file
from functions.rl.ev_dqn import EVAgent, train_agent, objective
from functions.rl.ev_env import EVEnv
from gym import spaces
import optuna


def main_dqn_function(track_data_interp, options):
    # Create the EV environment and agent
    #constants = Constants()
    #cost_function = CostFunction.ENERGY_EFFICIENCY  
    env = EVEnv(track_data_interp, options)
    # Extract dimensions
    if isinstance(env.observation_space, spaces.Box):
        input_dim = env.observation_space.shape[0]
    elif isinstance(env.observation_space, spaces.Discrete):
        input_dim = env.observation_space.n

    if isinstance(env.action_space, spaces.Box):
        output_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, spaces.Discrete):
        output_dim = env.action_space.n
    #input_dim = env.observation_space
    #output_dim = env.action_space
    agent = EVAgent(env, input_dim, output_dim)

    # Create an Optuna study object and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, env, agent, track_data_interp), n_trials=10)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # Train the agent using the best hyperparameters found by Optuna
    best_agent = EVAgent(input_dim, output_dim, lr=best_params['lr'], gamma=best_params['gamma'],
                            batch_size=best_params['batch_size'], capacity=best_params['capacity'])
    total_rewards, final_episode_states = train_agent(env, best_agent, track_data_interp, episodes=2) #episodes 10000
    return "EV energy model function executed"
