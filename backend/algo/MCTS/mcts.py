"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import backend.algo.MCTS.ctl_checking as ctl
import math



class MCTS:
    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.move = None
        self.children = dict()
        self.exploration_weight = exploration_weight
        self.breakdown_Q = defaultdict(lambda: [0, 0])
        self.invalid_actions = dict()



    def choose(self, node, get_children=False):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")
            return self.Q[n] / self.N[n]
        
        if get_children:
            return max(self.children[node], key=score), self.children[node]
        return max(self.children[node], key=score)
    


    def run_mcts(self, node, defined_request=None, defined_vehicle=None):
        "Make the tree one layer better for one iteration."
        path = self._select(node, defined_request, defined_vehicle)
        leaf = path[-1]
        self._expand(leaf)
        dc_r = self._rollout(leaf)
        self._backpropagate(path, dc_r[0]+dc_r[1], dc_r)



    def _select(self, node, defined_request=None, defined_vehicle=None):
        "Find an unexplored descendent of `node`"
        "Starting at root node, recursively select optimal child nodes until a leaf node L is reached"
        "If leaf is not reached, proceed to the expand step"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored: 
                n = unexplored.pop()
                path.append(n)
                return path
            for i in range(len(path)):
                try:
                    if path[i].theta[defined_request] != defined_vehicle:
                        return path
                except:
                    pass
            node = self._uct_select(node) 
    


    def _criteria_select(self, node, defined_request, defined_vehicle):
        assert all(n in self.children for n in self.children[node])
        for nd in self.children[node]:
            try:
                if nd.theta[defined_request] == defined_vehicle:
                    return node
            except:
                pass
        return None



    def _expand(self, node):
        "Expand a node by finding its children"
        "Update the `children` dict with the children of `node`"
        
        if node in self.children:
            return
        self.children[node], invalid_children = node.find_children()
        if len(invalid_children) > 0:
            self.invalid_actions[node] = invalid_children
        for i, nd in enumerate(self.children[node]):
            tree_node = ctl.CTLNode(i, nd)
            tree_node.evaluate(criterion="time")



    def _rollout(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        while True:
            if node.is_terminal():
                reward = node.reward(break_down=True)
                return reward
            rand_child = node.find_random_child()
            node = rand_child



    def _backpropagate(self, path, reward, dc_r):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            self.breakdown_Q[node][0] += dc_r[0]
            self.breakdown_Q[node][1] += dc_r[1]



    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])
        def uct(n):
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )
        return max(self.children[node], key=uct)



class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True