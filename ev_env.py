import numpy as np
import gym
from gym import spaces
#import torch
import pygame
import time
import random
#import ev_game
from functions.rl.Constant import CostFunction
from functions.rl.fcnEV import fcnEV_Class


class EVEnv(gym.Env):

    def __init__(self, track_data, options):
        super(EVEnv, self).__init__() ##First Done Mark
        
        #------------------------------------------------------Game Model Development------------------------------------------------------------------# 
        # To do
        #------------------------------------------------------Vehicle parameters------------------------------------------------------------------# 
        self.track_data = track_data
        self.options = options
        self.road_angle__rad = self.track_data['alpha']   
        self.distance = self.options['delta_s']
        self.start_flag = self.options['start_flag']
        self.stop_flag = len(track_data['v_max'])
        
        ##---------------------------------------------State-Action space init-----------------------------------------------------------------##
        # State 1: initial velocity limit, State 2: State of Charge limit, State 3: Time to reach destination limit
        self.SOC_min  = 0.2
        self.SOC_max  = 1 
        self.total_time_min__s = 0
        self.total_time_max__s = 1000
        
        # Action space: Torque u1 & State space: [speed x1, SOC x2, time x3]
        self.torque_lower__N_m = -700 # Nm
        self.torque_upper__N_m = 280  # Nm
        self.action_space = spaces.Box(low=self.torque_lower__N_m, high=self.torque_upper__N_m, shape=(1,), dtype=np.float32)
        
        # Initial states
        self.prev_state = [0.1, 0.2, 0.1]
        self.next_state = [0, 0, 0]
        self.current_step = 0
        self.done = False  
        self.old_score = 0
        self.reward = 0 
        self._update_observation_space() # Initialize observation space with initial limits
        self.fcnEV_instance = None

    def _update_observation_space(self):
        # Dynamically update observation space based on current step
        v_max_step__m_per_s = self.track_data['v_max'][self.current_step]
        v_min_step__m_per_s = self.track_data['v_min'][self.current_step]
        self.observation_space = spaces.Box(
            low=np.array([v_min_step__m_per_s, self.SOC_min, self.total_time_min__s]),
            high=np.array([v_max_step__m_per_s, self.SOC_max, self.total_time_max__s]),
            dtype=np.float32
        )

    def seed(self, seed=None):
        pass

    def reset(self): ##First Done Mark
        
        #--------------------- Reset the environment to initial state ---------------------## 
        # State 1: initial velocity init State 2: State of Charge init State 3: Time to reach destination limit
        v_init__m_per_s = 0.1 if self.start_flag else self.track_data['v_max'][0]    
        SOC_init = 20 
        time_init__s = 0
        self.total_time_segment_s = 0

        self.prev_state = [v_init__m_per_s, SOC_init, time_init__s]
        self.current_step = 0
        self.done = False
        self._update_observation_space()
        return self.prev_state

    def get_legal_actions(self): ##First Done Mark
        
        #----------------------------- to do (observation space init redundant) -------------------#
        print('------ Into legal action check --------')
        self._update_observation_space()
        aplha_step__rad_per_s = self.road_angle__rad[self.current_step]
        legal_actions = []
        legal_selections = np.linspace(self.torque_lower__N_m, self.torque_upper__N_m, num=1000) 

        for action in legal_selections:
            self.fcnEV_instance = fcnEV_Class(self.prev_state, action, aplha_step__rad_per_s, self.distance)
            if not self.fcnEV_instance.infeasible:
                legal_actions.append(action)
        
        #print(len(legal_actions))
        return legal_actions 

    def get_reward(self, fuel_consumed, total_fuel_consumed):
        
        #--------------------- Parameters for reward computation -------------------------# 
        self.reward = 0.0 # Initialize reward

        # 1. **Fuel Consumption:** Reward for minimizing fuel consumption
        fuel_penalty_scale = -0.5 # 1. **Fuel Consumption:** Reward for minimizing fuel consumption
        self.reward += fuel_penalty_scale * fuel_consumed

        # 2. **Velocity Changes:** Severe penalty for abrupt changes in velocity
        velocity_change_penalty_scale = -5.0  # Severe penalty for abrupt changes
        velocity_change = abs(self.prev_state[0] - self.next_state[0])
        if velocity_change > 5: # Define a threshold for abrupt changes
            self.reward += velocity_change_penalty_scale * velocity_change
        
        # 3. **Time Minimization:** Reward for minimizing time taken between points
        time_penalty_scale = -0.1  # Encourage faster travel
        time_taken = self.prev_state[2] - self.next_state[2]
        self.reward += time_penalty_scale * time_taken

        # 4. **SOC Constraints:** Heavy penalty if SOC is outside acceptable limits
        soc_penalty_scale = -10.0  # Heavy penalty for SOC violations
        if self.next_state[1] < self.SOC_min or self.next_state[1] > self.SOC_max:
            self.reward += soc_penalty_scale * abs(self.next_state[1] - np.clip(self.next_state[1], self.SOC_min, self.SOC_max))
    
        # 5. **Long-Term Goal:** Encourage reaching the destination with minimal time and fuel consumption
        if (self.done == 1):
            long_term_goal_reward = 100.0  # Bonus for reaching the destination
            total_time_penalty_scale = -0.05  # Penalty for overall time taken
            total_fuel_penalty_scale = -1.0  # Penalty for overall fuel consumption
            total_time_taken = self.total_time_segment_s

            self.reward += long_term_goal_reward
            self.reward += total_time_penalty_scale * total_time_taken
            self.reward += total_fuel_penalty_scale * total_fuel_consumed

        return self.reward 

    def step(self, action): ##First Done Mark
        #-------- Execute action, calculate next state from fcnEV and reward from get_reward -------#
        print('------ Into Step --------')
        self.fcnEV_instance = fcnEV_Class(self.prev_state, action, self.aplha_step__rad_per_s, self.distance, CostFunction.ENERGY_EFFICIENCY)
        self.next_state, self.total_time_segment_s = self.fcnEV_instance.get_state()
        fuel_consumed, total_fuel_consumed = self.fcnEV_instance.rewards_energy_efficiency_func()
        self.reward = self.get_reward(fuel_consumed, total_fuel_consumed)
        self.prev_state = self.next_state

        # Update the step
        self.current_step = self.current_step + 1 

        # Check if episode is done
        self.done = self.current_step >= self.stop_flag
        if (self.done == 1):
            self.current_step = 0
            print('------ Step Completed --------')
        
        # Return observation, reward, done flag, and any additional info
        return self.prev_state, self.reward, self.done
    

if __name__ == "__main__":
    env = EVEnv()
    observation = env.reset()
    # Perform a random action and observe the outcome
    #env.render()
    start_time = time.time() 
    while not done:
        #time.sleep(1)
        legal_actions = env.get_legal_actions()
        action = random.choice(legal_actions)
        observation, reward, done = env.step(action)
        #env.render()
        #if env.scoreboard_view_model.is_finished():
            #end_time = time.time() 
            #elapsed_time = end_time - start_time
            #start_time = time.time() 
            #print(f"Elapsed time: {elapsed_time} seconds")
            #print('Reward: ', reward)
            #env.yahtzee_game_view_model.handle_reset()
            

    