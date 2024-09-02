# Will want to load in environment
# 

import gym
import numpy as np
import time

class GameRunner:
    
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode=None)
        self.initial_state = self.env.reset()
        # print(dir(self.env))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def run(self):
        
        episodeNumber = 5
        timestep = 100
        
        for episode in range(episodeNumber):
            initial_state = self.env.reset()
            time_index = 1
            
            observations = []
            # for timeIndex in range(timestep):
            while not terminated:
                print(f'[{time_index}]')
                
                # Replace this with first method of solving problem
                random_action = self.env.action_space.sample()
                
                # interaction with env
                observation, reward, terminated, truncated, info = self.env.step(random_action)
                
                # Dealing with its actions
                observations.append(observation)
                time_index += 1
                
        self.env.close()