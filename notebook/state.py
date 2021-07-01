import time
import torch
import uuid

class State():
    # list of string
    def __init__(self, state):
        # state is type of mask Tensor with shape [28, 28]
        self.state = state
        self.id = uuid.uuid4()
        
    def __repr__(self):
        return str(self.state)
        
    def __eq__(self, other):
        return torch.equal(self.state, other.state)
    
    def __hash__(self):
        return hash(self.__repr__())
    
    def __len__(self):
        return len(self.state)
    
    def add(self, other):
        return State(torch.logical_or(self.state.bool(), other.state.bool()).float())
    
    def minus(self, other):
        return State(torch.logical_and(self.state.bool(), other.state.bool()).float())
    
    def get_id(self):
        return str(self.id)
    
    def get_state(self):
        return self.state