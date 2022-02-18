import numpy as np
from collections import defaultdict






def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
    """ updates the action-value function estimate using the most recent time step """
    return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

class Agent:

    def __init__(self, nA=6, epsilon=1, epsilon_decay=0.99, alpha=0.015, gamma=1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self._nA = nA
        self._Q = defaultdict(lambda: np.zeros(self._nA))
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._gamma = gamma
        self._alpha=alpha
        
    def _get_policy(self, state):
        p =np.ones(self._nA)*self._epsilon/self._nA
        p[np.argmax(self._Q[state])]=1-self._epsilon+self._epsilon/self._nA
        return p
    
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        p=self._get_policy(state)
        action = np.random.choice(np.arange(self._nA), p = p)

        return action
    

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        p= self._get_policy(next_state)

        self._Q[state][action] = update_Q(self._Q[state][action], p.dot(self._Q[next_state]), reward, self._alpha, self._gamma)
        if done:
            self._epsilon*=self._epsilon_decay
            self._epsilon=max(self._epsilon, 0.001)