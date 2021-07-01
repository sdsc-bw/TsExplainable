from abc import ABCMeta, abstractmethod
from mct import MC_node, MC_edge, MCFE_tree
from state import State

import torch

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

max_len = 20

class Agent(metaclass = ABCMeta):
    
    def run(self, episodes=None, min_episodes=0, n_edges=1, step_when_leaf_not_done=True):
        """Runs the episodes of a MCTS. Can analyze a number of the best edges of the root more closly.

        Keyword arguments:
        episodes -- number of episodes
        n_edges -- number of edges that are looked at more closly
        """
        print(f'- Start to tain agent, with:')
        print(f'    c: {self.c}')
        print(f'    max_depth: {self.max_depth}')
        print(f'    fsize: {self.game.fsize}')
        print(f'    Original ksize: {self.game.ksize}')
        print(f'    number actions: {len(self.game.action_space)}')
        if n_edges == 0:
            i = 0
            distrb = []
            while True:
                self.episode(self.root)
                if i % 1000 == 0:
                    self.logger.info('X'*70)
                    self.logger.info('Round:\t%d'%(i))
                    self.logger.info('X'*70)
                if i % self.root.get_num_edges() == 0 and i != 0:
                    curr_distrb = self.root.get_winrate_of_edges()
                    diff =  sum([abs(a-b) for a, b in zip(distrb, curr_distrb)])  / self.root.get_num_edges()
                    if diff < self.eps and i > min_episodes and len(distrb) > 0:
                        self.logger.info('Distribution stable after \t%d episodes'%(i))
                        break
                    distrb = curr_distrb
                i += 1
                if episodes is not None and i >= episodes:
                    break
        else:
            best_edges = []
            for j in range(n_edges):
                if j == 0:
                    curr_root = self.root
                    depth = 0
                    distrb = []
                else:
                    curr_root = best_edges[j].out_node
                    depth = 1
                    distrb = curr_root.get_winrate_of_edges()

                while depth < 10:
                    #for i in range(episodes):
                    i = 0
                    while True:
                        self.episode(curr_root)
                        if i % 1000 == 0:
                            self.logger.info('X'*70)
                            self.logger.info('Round:\t%d'%(i))
                            self.logger.info('X'*70)
                        if i % curr_root.get_num_edges() == 0 and i != 0:
                            curr_distrb = curr_root.get_winrate_of_edges()
                            diff =  sum([abs(a-b) for a, b in zip(distrb, curr_distrb)])  / self.root.get_num_edges()
                            if diff < self.eps and i > min_episodes and len(distrb) > 0:
                                self.logger.info('Distribution stable after \t%d episodes'%(i))
                                break
                            distrb = curr_distrb
                        i += 1
                        if episodes is not None and i >= episodes:
                            break
                    edges = curr_root.sort_edges_by_N()
                    curr_root = edges[0].out_node
                    if j == 0 and depth == 0:
                        best_edges = edges[1:(n_edges + 1)]
                    if curr_root.is_leaf() or curr_root.game_is_done:
                        break
                    distrb = curr_root.get_winrate_of_edges()
                    depth += 1

    @abstractmethod
    def episode(self, root):
        """Performs a episode of the MCTS. Starts the selection at the given root node.

        Keyword arguments:
        root -- node from where to start the selection
        """
        pass
    
    @abstractmethod
    def roll_out(self, leaf):
        """Performs a rollout of the MCTS for a given leaf. Returns the end node.

        Keyword arguments:
        leaf -- node from where to start the rollout
        """
        pass

    def get_results(self):
        """Returns the path and the leaf node of the path with the highest winrate.
        """
        return self.mct.selection_with_N()
    
    def get_data(self):
        """Returns the states and their action winrates with more than 1000 visits.
        """
        nodes = []
        states = []
        winrates = []
        actions = self.game.all_actions
        for node in self.mct.tree:
            winrate = []
            if not node.is_leaf() and node.N >= 100:
                state = State(node.state.state * self.game.figure)
                if state not in states:
                    states.append(state)
                    for action in self.game.all_actions:
                        edge = node.get_edge_with_action(action)
                        if edge == 0:
                            winrate.append(0.0)
                        else:
                            winrate.append(edge.get_winrate())
                    winrates.append(winrate)
        states = [s.state for s in states]
        return states, winrates
    
    @abstractmethod
    def get_best_path(self):
        """Returns rank and mask of the path with the highest winrate.
        """
        pass
    
    @abstractmethod
    def get_best_actions(self, masked_figure, n=5):
        """Returns the ranks and mask of the input state with the n best actions.

        Keyword arguments:
        masked_figure -- a masked state of the initial figure
        n -- number of actions to output
        """
        pass
        
    def _create_edges_for_leaf_and_evaluate(self, leaf):
        """Returns the possible edges and their values for the input leaf.

        Keyword arguments:
        leaf -- the input leaf 
        """
        state_leaf = leaf.state
        # get index of available_actions
        available_actions = self.game.get_available_actions(state_leaf)
        if len(available_actions) == 0:
            return [], []
        # get the next state through simulation
        values, is_dones, new_states = self.game.evaluate_actions_at_state(available_actions, state_leaf)
        edges = [MC_edge(action, leaf, MC_node(state, depth=leaf.depth + 1, game_is_done=is_done), self.c, value) for action, state, value, is_done in zip(available_actions, new_states, values, is_dones)]
        return edges, values

class Two_Player_Minus_Agent(Agent):
    
    def __init__(self, game, c=0.05, logger = None):
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
        self.game = game
        self.root_state = self.game.initial_state
        self.mct = MCFE_tree(self.root_state, logger = self.logger)
        self.root = self.mct.root
        #self.mct.add_actions(self.game.available_actions)
        self.num_episode = 0
        self.c = c
        self.eps = 0.00001
        self.max_depth = self.game.max_depth
        self.best_record = {'node': self.root, 'depth': 10000}
        self.root_state_player0 = self.root_state
        self.root_state_player1 = self.root_state
    
    #def episode(self, root, player):
    #    self.episode1(root, player)
        
    def episode(self, root, player):
        self.num_episode += 1
        node, path = self.mct.selection(root=root)
        self.mct.add_node_to_tree(node)
        if not node.game_is_done:
            # didn't reach the maximum level, expand
            # evaluate node state
            
            self.logger.debug(f'- start the expansion process')
            edges, values = self._create_edges_for_leaf_and_evaluate(node, player)
            if len(edges) != 0:
                # expansion with ts
                expanded_edge = self.mct.expansion(node, edges, values)
                path.append(expanded_edge)
                node = expanded_edge.get_out_node()
                
            self.logger.debug(f'+ end the expansion process, {len(edges)} nodes are added to the tree')
        if not node.game_is_done and node.depth< self.max_depth:
            reward = self.roll_out(node, player)
        elif node.depth >= self.max_depth:
            reward = 0
        else:
            if self.best_record['depth'] > node.depth:# and player == 0:
                self.best_record['node'] = node
                self.best_record['depth'] = node.depth
            reward = self.game.get_reward_with_node(node)
        self.mct.back_fill(reward, path)
    
    def get_best_path(self):
        _, path = self.mct.selection_with_N()
        n = len(path)
        ranks = np.zeros(self.game.figure.shape)
        mask = np.zeros(self.game.figure.shape)
        if self.game.start == self.game.target:
            factor = 1
        else:
            factor = -1
        for i in range(1, n + 1):
            action = self.game.normalize(path[i - 1].action.action)
            ranks += factor * i * action
            mask += action
        return ranks, mask
    
    def get_best_actions(self, masked_figure, n=5):
        if n == 0:
            return  np.zeros(self.game.figure.shape), np.zeros(self.game.figure.shape)
        state = self.game.normalize(self.game.figure - masked_figure)
        # get node with highest N
        best_node = None
        for node in self.mct.tree:
            if torch.equal(masked_figure, node.state.state * self.game.figure):
                if best_node is None or best_node.N < node.N:
                    best_node = node            
        if best_node is None:
            raise KeyError('State not in tree.')
        edges = best_node.sort_edges_by_winrate()[:n]
        edges = best_node.edges[:n]
        ranks = np.zeros(self.game.figure.shape)
        mask = np.zeros(self.game.figure.shape)
        if self.game.start == self.game.target:
            factor = 1
        else:
            factor = -1
        for i in range(len(edges)):
            action = self.game.normalize(edges[i].action.action)
            ranks += factor * (i + 1) * action
            mask += action
        return ranks, mask 
    
    def refine_action(self, actions):
        size = round(self.game.ksize/2)
        list_actions = self.game.reset_action_space_with_selected_actions(actions, size )
        return [list_actions], size
    
    def refine_action_sequential(self, actions):
        size = round(self.game.ksize/2)
        list_actions = self.game.reset_action_space_with_selected_actions_sequential(actions, size)
        return list_actions, size
    
    def roll_out(self, leaf, player):
        #print('- start to roll out the give node')
        state_leaf = leaf.state
        # change continue to parallel in pc
        available_actions = self.game.get_available_actions(state_leaf)
        num_selection = min(self.max_depth - leaf.depth, len(available_actions))
        if num_selection == 0:
            print('    no action available')
            return 0
        states = random.sample(available_actions, num_selection)
        # cumsum of actions
        
        states[0] = leaf.state.minus(State(states[0]))
        
        #print(f'    number of missing action is: {len(states)}')
        for i in np.arange(1,len(states)):
            states[i] = states[i-1].minus(State(states[i]))
        is_dones, probs = self.game.is_done(states, player)
        for i, j in enumerate(is_dones): # get the first end game
            if j == True:
                break
        depth = leaf.depth + i
        prob = probs[i]
        reward = self.game.get_reward(depth, prob, player)
        self.logger.debug(f'+ The roll out finish at depth {depth} with probability {prob}, reward of the leaf node is {reward}')
        return reward
    
    def run(self, episodes=None):
        """Runs the episodes of a MCTS. Can analyze a number of the best edges of the root more closly.

        Keyword arguments:
        episodes -- number of episodes
        n_edges -- number of edges that are looked at more closly
        """
        print(f'- Start to tain agent, with:')
        print(f'    c: {self.c}')
        print(f'    max_depth: {self.max_depth}')
        print(f'    fsize: {self.game.fsize}')
        print(f'    Original ksize: {self.game.ksize}')
        print(f'    number actions: {len(self.game.action_space)}')
        list_actions = [self.game.action_space]
        flag = True
        while self.game.ksize > 5:
            print(f'---------------------------------------------------------------------------------')
            print(f'The current ksize is {self.game.ksize}')
            if flag:
                print('- Run first level episode')
                [self.episode(self.root) for i in range(50)]
                print('+ End with first level')
                print(self.root.get_infor_of_edges())
                tmp_actions = self._choose_actions_to_refine_sequential()
                if len(tmp_actions) == 4:
                    print('+ Black box model always predict right, No shortest path found, return 0')
                    return list_actions
                new_list_actions, size = self.refine_action_sequential(tmp_actions)
                flag = False
            else:
                print(f'- Run lower level episode, the length of list_actions is {len(list_actions[0])}')
                self.game.set_action_space(list_actions[0])
                self.game.reset_max_depth()
                self.max_depth = self.game.max_depth
                initial_state = self.game.initial_state
                self.mct = MCFE_tree(initial_state, logger = self.logger)  # necessary?
                self.root = self.mct.root
                print(f'    - sum of the mask is {self.root.state.state.sum()}')
                self.best_record = {'node': self.root, 'depth': 10000}
                print(f'    - Start to build tree')
                min_epi = 500
                counter = 0
                while True:
                    self.episode(self.root)     # run episode
                    if counter >= min_epi:
                        if self.best_record['depth'] >1000:
                            min_epi += 100
                        else:
                            break
                    counter += 1
                print(f'    - End with tree generation')
                print(self.root.get_infor_of_edges())
                tmp_actions = self._choose_actions_to_refine()
                tmp, size = self.refine_action(tmp_actions)
                new_list_actions = tmp # refine 
                print('+ End with lower level episode')
            self.game.set_ksize(size)
            list_actions = new_list_actions
            if len(list_actions) == 0:
                print('+ End running, all factor explored')
                break
        return self.mct.selection_with_N()
            
    def run_sequential(self):
        node, path = self.generate_mct_sequential(0)
        # reset game for player
        self.root_state_player1 = node.state
        self.game.reset_ksize()
        self.game.reset_action_space()
        prob = self.game.get_prob_with_node(node)
        self.game.set_init_prob_player1(prob)
        print(f'Init probablity for player2: {prob}')
        node2, path2 = self.generate_mct_sequential(1)
        return node, node2, path, path2
    
    def generate_mct_sequential(self, player):
        # sequential method会影响最后的效果，最终通过选中的node来体现各个feature的价值
        list_actions = [self.game.action_space]
        flag = True
        if self.game.get_initial_score() < 0.5 and player == 0:
            return self.mct.selection_with_N()
        while self.game.ksize > 20:
            print(f'---------------------------------------------------------------------------------')
            print(f'The current ksize is {self.game.ksize}')
            if flag:
                if player == 0:
                    self.mct = MCFE_tree(self.root_state_player0, logger = self.logger)  # necessary?
                else:
                    self.mct = MCFE_tree(self.root_state_player1, logger = self.logger)  # necessary?
                self.root = self.mct.root
                print('- Run first level episode')
                [self.episode(self.root, player) for i in range(50)]
                print('+ End with first level')
                print(self.root.get_infor_of_edges())
                tmp_actions = self._choose_actions_to_refine_sequential()
                if len(tmp_actions) == 4:
                    print('+ Black box model always predict right, No shortest path found, return 0')
                    return self.mct.selection_with_N()
                new_list_actions, size = self.refine_action_sequential(tmp_actions)
                flag = False
            else:
                print(f'- Run lower level episode, the length of list_actions is {len(list_actions)}')
                new_list_actions = []
                for i in np.arange(len(list_actions)-1, -1, -1):
                    print(f'    process the {i}th item in the list_actions')
                    self.game.set_action_space(list_actions[i])
                    self.game.reset_max_depth()
                    self.max_depth = self.game.max_depth
                    initial_state = self.game.initial_state
                    # get root 从最不重要的开始，逐渐探索
                    for actions in list_actions[:i]:
                        for action in actions:
                            initial_state = initial_state.minus(State(action))
                    for actions in new_list_actions: # 新的小家伙也要盖掉
                        #print(len(actions))
                        for action in actions:
                            initial_state = initial_state.minus(State(action))
                    #plt.imshow(initial_state.state)
                    if player == 0:
                        self.mct = MCFE_tree(self.root_state_player0.minus(initial_state), logger = self.logger)  # necessary?
                    else:
                        self.mct = MCFE_tree(self.root_state_player1.minus(initial_state), logger = self.logger)
                    self.root = self.mct.root
                    print(f'    - sum of the mask is {self.root.state.state.sum()}')
                    self.best_record = {'node': self.root, 'depth': 10000}
                    print(f'    - Start to build tree')
                    
                    [self.episode(self.root, player) for i in range(50)]     # run 50 episode
                    print(f'    - End with tree generation')
                    if len(self.root.edges) != 0:
                        print(self.root.get_infor_of_edges())
                    tmp_actions = self._choose_actions_to_refine_sequential()
                    #if len(tmp_actions) == 4:
                    #    print('     refine current action set is not necessary, since all factor are important')
                    #    continue
                    if len(tmp_actions) > 0:
                        tmp, size = self.refine_action_sequential(tmp_actions)
                        tmp.extend(new_list_actions)
                        new_list_actions = tmp # refine 
                print('+ End with lower level episode')
            self.game.set_ksize(size)
            list_actions = new_list_actions
            if len(list_actions) == 0:
                print('+ End running, all factor explored')
                break
        return self.mct.selection_with_N()
        
            
    def _choose_actions_to_refine(self):
        # TODO !!!!!!
        #  可以细化的条件是，去掉这个大框能导致分类失误，所以如何确定细化的地方是个学问，还有细化后的目的是什么也是个问题
        # 同时还有废掉ksize
        return self._choose_actions_to_refine_sequential()
    
    def _choose_actions_to_refine_sequential(self):
        # 这种方法要考虑到action重要性的顺序，所以不能像上面一样直接用best score
        print(f'- Choose action that need to refine, with the winrate selection')
        _, path = self.mct.selection_with_N()
        actions = [i.action for i in path]
        print(f'+ End with action selection, list length: {len(actions)}')
        return actions
        
    
    def _create_edges_for_leaf_and_evaluate(self, leaf, player):
        """Returns the possible edges and their values for the input leaf.

        Keyword arguments:
        leaf -- the input leaf 
        """
        state_leaf = leaf.state
        # get index of available_actions
        available_actions = self.game.get_available_actions(state_leaf)
        if len(available_actions) == 0:
            return [], []
        # get the next state through simulation
        values, is_dones, new_states = self.game.evaluate_actions_at_state(available_actions, state_leaf, player)
        edges = [MC_edge(action, leaf, MC_node(state, depth=leaf.depth + 1, game_is_done=is_done), self.c, value) for action, state, value, is_done in zip(available_actions, new_states, values, is_dones)]
        return edges, values
    
    def _locate_mask(a):
        flag = False
        i1 = 0
        j1 = 0
        for i in range(a.shape[0]):
            if flag:
                j2 = j
                break
            for j in range(a.shape[1]):
                if a[i,j] == 0 and not flag:
                    i1 = i
                    j1 = j
                    flag = True
                if a[i,j] == 1 and flag:
                    j2 = j
                    break
        return i1, j1, i1+j2-j1, j2