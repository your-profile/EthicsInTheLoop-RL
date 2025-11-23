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
        self.decay = decay 

        # Total states= 5408
        num_states = 26*26*2*2*2
        self.qtable = pd.DataFrame(np.random.rand(num_states, action_space), columns=list(range(action_space)))

    def trans(self, state, granularity=1.0, verbose=False):

        shopping_list = set(state['observation']['players'][0]['shopping_list'])
        selected_items = []
        purchased_items = []

        if len(state['observation']['baskets']) > 0:
            basket_list = set(state['observation']['baskets'][0]['contents'])
            purchased_list = set(state['observation']['baskets'][0]['purchased_contents'])
            selected_items = shopping_list.difference(basket_list)
            purchased_items = shopping_list.difference(purchased_list)


        # You should design a function to transform the huge state into a learnable state for the agent
        # It should be simple but also contains enough information for the agent to learn
        player_x = int(state['observation']['players'][0]['position'][0]*granularity)
        player_y = int(state['observation']['players'][0]['position'][1]*granularity)
        has_basket = int(state['observation']['players'][0]['curr_basket'] + 1)
        has_items = int(len(list(selected_items)))
        has_checkout = int(len(list(purchased_items)))

        if has_basket >= 1:
            if verbose:
                print("Has a basket!")
            has_basket = 1

            if has_items ==0:
                if verbose:
                    print("Has all items!")
                has_items = 1

            if has_checkout == 0:
                if verbose:
                    print("Purchased all items!")
                has_checkout = 1

        #encoding: ((((x,y)*2 + cart)*2 + items)*2 + checkout)
        idx = ((((player_x*26 + player_y)*2 + has_basket)*2 + has_items)*2 + has_checkout)

        return idx

    def learning(self, action, rwd, state, next_state):
        td_error = rwd + self.gamma * np.max(self.qtable.loc[next_state]) - self.qtable.loc[state, action]
        update = self.alpha * td_error
        self.qtable.loc[state, action] = self.qtable.loc[state, action] + update

    def priming(self, action, prime_num, state, next_state):
        self.qtable.loc[state, action] += prime_num


    def choose_action(self, state):
        rand = np.random.rand()

        if rand < self.epsilon:
            action = np.random.choice(range(self.action_space))
        else:
            action = np.argmax(self.qtable.loc[state])

        return action