import time
import torch
import numpy as np
from numpy import log as ln

class Action():
    
    # list of string
    def __init__(self, action):
        # action is type of mask Tensor with shape [28, 28]
        self.action = action
        self.parent_nodes = []
        self.N = 0
        self.W = 0
        
    def __repr__(self):
        return str(self.action)
        
    def __eq__(self, other):
        return torch.equal(self.action, other.action)
    
    def __hash__(self):
        return hash(self.__repr__())
    
    def __len__(self):
        return len(self.action)
    
    def get_action(self):
        return self.action