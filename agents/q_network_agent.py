import numpy as np

class QNetwork_Agent():
    
    def __init__(self, action_space_size: int = None, observation_space_size: int = None):
        '''
        
        '''
        self.q_network = self.create_q_network(action_space_size, observation_space_size)
        
    def create_q_network(self, action_space_size, observation_space_size):
        
        network = np.zeros(action_space_size, observation_space_size, dtype=float)
        return network
    
    def take_action(self, observation):
        pass
    
    def train_network(self):
        pass