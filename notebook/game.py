from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import logging
import random
import torch
import torch.nn.functional as F

from state import State
from mct import MC_node


class Game(metaclass = ABCMeta):
    
    @abstractmethod
    def get_available_actions(self, state):
        """Returns the available actions for the input state.

        Keyword arguments:
        state -- the input state
        """
        pass
    
    @abstractmethod
    def _get_initial_state(self):
        """Returns the initial state.
        """
        pass
    
    @abstractmethod
    def get_reward(self, node):
        """Returns the reward for the input node.

        Keyword arguments:
        state -- the input node
        """
        pass
    
    @abstractmethod
    def is_done(self, state):
        """Returns if the game is done for the input state. 

        Keyword arguments:
        state -- the input state
        """
        pass
    
    @abstractmethod
    def simulate_action(self, state, action):
        """Applies the input action on the input state and returns the resulting state.

        Keyword arguments:
        state -- the input state
        action -- the input action
        """
        pass
    @abstractmethod
    def evaluate_actions_at_state(self, action, state):
        """Returns the value of the input action for the input.
        
        Keyword arguments:
        state -- the input state
        action -- the input action
        """
        pass
    
    @abstractmethod
    def normalize(self, tensor):
        """Normalizes the tensor in order to create a mask based on the game.

        Keyword arguments:
        tensor -- the input tensor
        """
        pass
    
    @abstractmethod
    def _generate_available_actions(self):
        """Creates and returns all available actions.
        """
        pass

                
class Two_Player_Minus_Game(Game):
    
    def __init__(self, figure, black_box, target, network_0=None, network_1=None, rollout_ready=True, fsize=28, ksize=3, ratio=0.0, threshold=0.1, threshold2 = 0.05, max_depth = 10, logger = None):
        """
        figure: input instance after process
        black_box:
        target: int # change this to the class of the input instance
        network_0: network for the first player
        network_1: network for the second player
        rollout_ready: ???
        fsize: width and length of the input instance (28)
        ksize: kernel size, size of super pixel (3)
        ratio: weight of the probability in reward (0.0)
        threshold: ?? (0.0)
        """
        ######## set logger #########
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
            
        ######## parameters initalization ######
        self.figure = figure # target figure to analyse
        #self.start = int(F.softmax(black_box(self.figure.unsqueeze(0)), dim=1).argmax().item()) # get the init prediction of the player one
        self.target = target 
        self.black_box = black_box # input: Tensor [n, 1, width, heigth] output: Tensor [n, num_classes]
        self.fsize = fsize
        self.original_ksize = ksize
        self.ksize = ksize
        self.initial_state = self._get_initial_state() # figure with all one with size [fsize, fsize]
        self.action_space = self._generate_available_actions() # depends on fsize and ksize
        self.network_0 = network_0
        self.network_1 = network_1
        self.max_depth = len(self.action_space)
        self.offset = 1 # ??
        self.ratio = ratio 
        self.threshold = max(self.get_prob_with_node(MC_node(self.initial_state)) - threshold, 0.5) # ?? 
        self.threshold2 = threshold2
        self.rollout_ready = rollout_ready
        self.init_prob_player1 = 0 # will be reset after the running of the first player.
        
    def evaluate_actions_at_state(self, actions, state, player):
        values, is_done, new_states = self.get_prediction_change_and_test_if_it_is_done(actions, state, player)
        return values, is_done, new_states
    
    def get_available_actions(self, state_leaf):
        actions = []
        for i in self.action_space:
            if state_leaf.state.sum() - (state_leaf.state * i).sum() > self.ksize*self.ksize/2:
                actions.append(i)
        return actions
    
    def get_prediction_change_and_test_if_it_is_done(self, actions, state, player):
        
        #print("- get prediction change and test if it's done")
        # 比较take action前后在概率上的变化，并以此为反馈。
        new_states = [self.simulate_action(state, action) for action in actions] # why add? 所有可能的下代, change to minus here
        inp1 = state.state * self.figure # leaf masked figure [fsize, fsize] X [3, fsize, fsize]
        inp1 = inp1.unsqueeze(0) # only one item, so use unsqueeze
        inp2 = [i.state * self.figure for i in new_states] # expanded figure
        inp2 = torch.stack(inp2)
        inp = torch.cat([inp1, inp2]) # the first one is the leaf and the other are expanded nodes
        
        #print(f"    shape of inp is:{inp.shape}")
        self.black_box.eval()
        with torch.no_grad(): # get score for the old and new figure
            out = F.softmax(self.black_box(inp), dim=1)#[:, self.target].data.numpy()
        #preds = out[1:].argmax(dim=1).data.numpy()
        
        prob1 = np.repeat(out[0][self.target], out.shape[0]-1, axis = 0)
        prob2 = out[1:,self.target]
        if player == 0:
            #is_done = [i != self.target for i in preds]
            is_done = out[1:, self.target] < self.threshold     # 预测概率小于0.9 即结束游戏。这个得根据游戏和模型决定 
        else:
            #is_done = [i == self.target for i in preds]  #  当预测提高的时候结束游戏
            is_done = (prob2 - self.init_prob_player1)> self.threshold2
        #print(f"    prob1 is: {prob1}")
        #print(f"    prob2 is: {prob2}")
        if player == 0: ## ????
            values = prob1 - prob2
        elif player == 1:
            values = prob2 - prob1
        return values, is_done, new_states
    
    def get_prob_with_node(self, node):
        pred_t = 0.0
        state = node.state.state * self.figure
        pred_t = F.softmax(self.black_box(state.unsqueeze(0)), dim=1).squeeze()[self.target].data.numpy()
        return pred_t
    
    def get_initial_score(self):
        return self.get_prob_with_node(MC_node(self.initial_state))
    
    def get_reward_with_node(self, node):
        # return the ratio of mask
        #self.logger.info('Get reward for current state')
        # reward shortest path
        pred_t = 0.0
        if self.ratio > 0.0:
            state = node.state.state * self.figure
            pred_t = F.softmax(self.black_box(state.unsqueeze(0)), dim=1).squeeze()[self.target].data.numpy()
        reward = (1 - self.ratio) * max(1 - (node.depth - self.offset) / self.max_depth, 0) + self.ratio * pred_t
        #print(node.depth)
        #self.logger.info('Reward of current state: %f'%(reward))
        return reward
    
    def get_reward(self, depth, prob, player):
        pred_t = prob
        if player == 0:
            reward = (1 - self.ratio) * max(1 - (depth - self.offset) / self.max_depth, 0) + self.ratio * (1- pred_t)
        else:
            reward = (1 - self.ratio) * max(1 - (depth - self.offset) / self.max_depth, 0) + self.ratio * (pred_t)
        #self.logger.info('Reward of current state: %f'%(reward))
        return reward
    
    def is_done(self, states, player): 
        """
        take multiple states as input and output the improvement and judge if the game is done
        stetes: list of tensor
        """
        inp = [i.state * self.figure for i in states]
        inp = torch.stack(inp)
        with torch.no_grad(): # get score for the old and new figure
            out = F.softmax(self.black_box(inp), dim=1)#[:, self.target].data.numpy()
        preds = out.argmax(dim=1).data.numpy()
        probs = out[:,self.target]
        if player == 0:
            #is_done = [i != self.target for i in preds]
            is_done = out[:, self.target] < self.threshold
        else:
            #is_done = [i == self.target for i in preds]
            print(probs, self.init_prob_player1)
            is_done = (probs - self.init_prob_player1) > self.threshold2
        return is_done, probs
    
    def normalize(self, tensor):
        return np.logical_xor(np.array(tensor), np.ones(self.figure.shape), dtype=float).astype(float)     
    
    def reset_action_space(self):
        self.action_space = self._generate_available_actions()
        return 1
    
    def reset_action_space_with_selected_actions(self, actions, size):
        # 思考有哪些内容应该跟着改的：
        # 思考是否有必要全部的选中的动作的细分，因为选中的动作是目前导致误判的最小动作集
        # 思考针对动作是否可以挨个细分，或者减小细分的速度
        # 思考应该细分几次，
        # Forget to change the max depth after refine....
        print(f'- Refine grid to the chosen actions, the size of the original grid is {self.ksize}')
        if size == 0:
            print('    skize is small, no need to refine the grid, break the process')
            return 0
        action_space = []
        for action in actions:
            action_space.extend(self._refine_actions_in_mask(action, size))
        print(f'+ Finish refining, new grid size is: {size}, number of new actions is: {len(action_space)}')
        return action_space
    
    def reset_action_space_with_selected_actions_sequential(self, actions, size):
        # 顺序处理每个action，当处理第一个时，其他还没被处理过的维持全黑
        # 这个是要改变root的，和前面的不一样
        print(f'- Refine grid to the chosen actions, the size of the original grid is {self.ksize}')
        if size == 0:
            print('    skize is small, no need to refine the grid, break the process')
            return 0
        list_actions = []
        for action in actions:
            list_actions.append(self._refine_actions_in_mask_sequential(action, size))
        print(f'+ Finish refining, new grid size is: {size}, length of generated list_action is: {len(list_actions)}')
        return list_actions
    
    def reset_ksize(self):
        self.ksize = self.original_ksize
        return 1
    
    def reset_max_depth(self):
        self.max_depth = len(self.action_space)
        return 1
    
    def set_action_space(self, action_space):
        self.action_space = action_space
        return 1
    
    def set_init_prob_player1(self, value):
        self.init_prob_player1 = value
        return 1
    
    def set_ksize(self, size):
        self.ksize = size
        return 1
    
    def set_max_depth(self, max_depth):
        self.max_depth = max_depth
        return 1
    
    def simulate_action(self, state, action):
        return state.minus(State(action))
    
    def _generate_available_actions(self):
        all_actions = []
        #for i in np.arange(0, self.fsize - self.ksize + 1, self.ksize):
        #    for j in range(0, self.fsize - self.ksize + 1, self.ksize):
        for i in np.arange(0, self.fsize - 1, self.original_ksize):
            for j in range(0, self.fsize - 1, self.original_ksize):
                mask = torch.ones(self.fsize, self.fsize)
                mask[i:i+self.original_ksize, j:j+self.original_ksize] = 0
                all_actions.append(mask)
        return all_actions 
    
    def _get_initial_state(self):
        return State(torch.ones(self.fsize, self.fsize))
    
    def _refine_actions_in_mask(self, a, size):
        return self._refine_actions_in_mask_sequential(a, size)
    
    def _refine_actions_in_mask_sequential(self, a, size):
        # 这种方法有问题，不能看出哪个action是重要的
        for i in np.arange(0, self.fsize - self.ksize + 1, self.ksize):
            for j in range(a.shape[1]):
                if a[i,j] == 0 and (a[i:i+size, j:j+size]).sum() ==0:
                    actions = []
                    for m in np.arange(i, i+self.ksize, size):
                        for n in np.arange(j, j+self.ksize, size):
                            new_mask = torch.ones(self.fsize, self.fsize)
                            new_mask[m:m+size, n:n+size] = 0
                            actions.append(new_mask)
                    return actions

    

