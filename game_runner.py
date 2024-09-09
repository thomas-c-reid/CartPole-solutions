# Will want to load in environment
# 

import gym
import numpy as np
import time
from agents.q_table_agent import QTable_Agent

class GameRunner:
    
    """
    A class to initialize and manage the CartPole-v1 environment using OpenAI's Gym.

    Attributes:
    -----------
    env : gym.Env
        The CartPole-v1 environment instance.
    initial_state : object
        The initial state of the environment after reset.
    action_space : gym.Space
        The available actions the agent can take (discrete: left or right).
    observation_space : gym.Space
        The continuous state space of the environment, representing the cart position,
        cart velocity, pole angle, and pole angular velocity.
    """
    
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode=None)
        self.initial_state = self.env.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        q_table_params = self.load_params()
        self.QTable_agent = QTable_Agent(self.env, **q_table_params)
        
    def load_params(self):
        # In future use yaml file
        return {
            'epsilon': 0.1,
            'learning_rate': 0.005,
            'episodes': 10000
        }
    
    def train(self):
        
        episode_length = np.zeros(self.QTable_agent.episodes)
                
        for episode in range(self.QTable_agent.episodes):
            state = self.env.reset()
            time_index = 1
            
            terminated = False
            # for timeIndex in range(timestep):
            while not terminated:
                
                action = self.QTable_agent.pick_action(state)
                # interaction with env
                state, reward, terminated, truncated, info = self.env.step(action)
                
                if terminated:
                    episode_length[episode] = time_index
                
                # Dealing with its actions
                time_index += 1
                
                self.QTable_agent.update_q_table(reward, state)
                
        print(episode_length)
        self.env.close()
        
        
    def visualize_trained_agent(self, num_episodes=5):
        """ Run the trained agent and visualize the gameplay. """
        self.env = gym.make('CartPole-v1', render_mode='human')
        
        for episode in range(num_episodes):
            state = self.env.reset()
            terminated = False
            time_index = 0
            while not terminated:
                
                # Render the environment (shows the game visually)
                self.env.render()
                
                # Get the action from the trained agent (using the greedy policy)
                action = self.QTable_agent.pick_action(state, mode='BEST')
                
                # Perform the action in the environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                # Move to the next state
                state = next_state
                time_index += 1

            print(f"Episode {episode + 1} finished after {time_index} timesteps")
        self.env.close()