import numpy as np
import random
from gym import Env

class QTable_Agent():
    
    def __init__(self, env: Env = None, epsilon: float = 0.05, learning_rate: float = 0.001,
                 episodes: int = 60000, num_bins: int = 30):
        '''
        
        '''
        self.observation_space = env.observation_space
        self.low = env.observation_space.low
        self.high = env.observation_space.high
        self.action_space = env.action_space
        self.env = env
        self.num_bins = num_bins
        
        self.bins = [np.linspace(low, high, num_bins) for low, high in zip(self.low, self.high)]
        self.q_table = self.create_q_table(self.observation_space, self.action_space)
        
        self.epsilon = epsilon
        self.lr = learning_rate # This should maybe be higher
        self.episodes = episodes
        
        print('QTable Agent Initialised')
        
        
    def create_q_table(self, observation_space, action_space):
        
        q_table = np.zeros(([self.num_bins] * observation_space.shape[0] + [action_space.n]))
        
        return q_table
               
    def pick_action(self, state, mode='TRAIN'):
                
        state = self.make_state_discrete(state)
        
        if mode == 'TRAIN':
            action = self.epsilon_greedy_policy(state)
        elif mode == 'RANDOM':
            action = self.env.action_space.sample()
        elif mode == 'BEST':
            action = self.greedy_policy(state)
            
        # print('action: ', action)
            
        return action
               
    def greedy_policy(self, state):
        q_values = self.q_table[state]
        return np.argmax(q_values)
    
    def epsilon_greedy_policy(self, state):
                
        self.state_ = state
        p = random.random()
        
        if p > self.epsilon:
            action = self.greedy_policy(state)
        else:
            action = self.env.action_space.sample()
        
        return action

    def update_q_table(self, reward, next_state):
        """ 
        we recieve reward for previous state and next state
        """
        next_state = self.make_state_discrete(next_state)
        # now need to convert it to bins
        
        next_state_q_value = self.q_table[next_state]
        
        self.q_table[self.state_] = self.q_table[self.state_] + self.lr*(reward + next_state_q_value - self.q_table[self.state_])
        pass
    
    # def normalize(self, value, min_value, max_value):
        
    #     print('     . . .'*15)
    #     print(value - min_value)
    #     print(max_value - min_value)
    #     print('     . . .'*15)
        
    #     return (value - min_value) / (max_value - min_value)
    
    def normalize(self, value, min_value, max_value):
        # Handle cases where the range is infinite or very large
        if np.isinf(min_value) or np.isinf(max_value):
            return 0.5  # Default to the middle of the range if it's infinite

        # Handle extremely small ranges to avoid division by zero or near-zero
        if max_value - min_value == 0 or max_value - min_value < 1e-6:
            return 0.5  # Default to the middle if the range is too small

        # Normalization in the standard case
        return (value - min_value) / (max_value - min_value)
        
    def make_state_discrete(self, state):
                
        if type(state) == tuple:
            state = state[0]
        
        # normalize state
        print('*'*50)
        print('low', self.low)
        print('high', self.high)
        print(state)
        # state = [self.normalize(val, low_val, high_val) for val, low_val, high_val in zip(state, self.low, self.high)]    
        
        for idx, val in enumerate(state):
            print(val)
            print(low := self.low[idx])
            print(high := self.high[idx])
            print(self.normalize(val, low, high))
            print('-'*10)
            
        # print(state)
        print('*'*50)
        
        # splitting into bins 
        state = np.array([np.digitize(val, self.bins[idx]) -1 for idx, val in enumerate(state)])            
                
        return tuple(state)