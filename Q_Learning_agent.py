import numpy as np
import pandas as pd
# import torch.nn.functional as F
# import json  # Import json for dictionary serialization

class QLAgent:
    # here are some default parameters, you can use different ones
    def __init__(self, action_space, alpha=0.5, gamma=0.8, epsilon=0.1, mini_epsilon=0.01, decay=0.999):
        self.action_space = action_space 
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay               # value to decay the epsilon over time
        self.qtable = pd.DataFrame(np.random.rand(10000, 7), columns=[i for i in range(self.action_space)])

    
    def trans(self, state, granularity=1.0):
        # You should design a function to transform the huge state into a learnable state for the agent
        # It should be simple but also contains enough information for the agent to learn
        player_x = int(state['observation']['players'][0]['position'][0]*granularity)
        player_y = int(state['observation']['players'][0]['position'][1]*granularity)
        carts_has = int(len(state['observation']['carts']))

        if carts_has >= 2:
            carts_has = 2

        height = int(25 * granularity)
        width = int(20 * granularity)
        
        state_num = (player_x*width + player_y)*height + carts_has

        return state_num
        
        
    def learning(self, action, rwd, state, next_state):
        self.qtable.loc[state, action] = self.qtable.loc[state, action] + self.alpha * (rwd + self.gamma * np.max(self.qtable.loc[next_state]) - self.qtable.loc[state, action]
    )

        

    def choose_action(self, state):
        rand = np.random.rand()

        if rand < self.epsilon:
            action = np.random.choice(range(self.action_space))#, p = [0.18, 0.18, 0.18, 0.18, 0.18, 0.00, 0.1]) #['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'INTERACT', 'TOGGLE_CART']
        else:
            action = np.argmax(self.qtable.loc[state])

        return action