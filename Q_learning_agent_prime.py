import numpy as np
import pandas as pd
# import torch.nn.functional as F
# import json  # Import json for dictionary serialization

class QLAgent:
    # here are some default parameters, you can use different ones
    def __init__(self, action_space, alpha=0.5, gamma=0.8, epsilon=0.1, mini_epsilon=0.01, decay=0.999, granularity = 4.0, height = 26, width = 26):
        self.action_space = action_space 
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay 
        self.height = int(height*granularity)
        self.width = int(width*granularity)
        self.granularity = granularity

        # Total states= 5408
        num_states = self.height*self.width*2*2*2
        self.qtable = pd.DataFrame(np.random.rand(num_states, action_space), columns=list(range(action_space)))

    def trans(self, state, verbose=False, return_checks=False):

        shopping_list = set(state['observation']['players'][0]['shopping_list'])
        selected_items = []
        purchased_items = []

        if len(state['observation']['baskets']) > 0:
            basket_list = set(state['observation']['baskets'][0]['contents'])
            purchased_list = set(state['observation']['baskets'][0]['purchased_contents'])
            selected_items = shopping_list.difference(basket_list)
            purchased_items = shopping_list.difference(purchased_list)
            print(f"Shopping List: {shopping_list}, In Basket: {basket_list}, Purchased: {purchased_list}")
            print(f"Purchased Items: {purchased_items}, Selected Items: {selected_items}")



        # You should design a function to transform the huge state into a learnable state for the agent
        # It should be simple but also contains enough information for the agent to learn
        # print(state['observation']['players'][0]['position'])
        player_x = round(state['observation']['players'][0]['position'][0]*self.granularity)
        player_y = round(state['observation']['players'][0]['position'][1]*self.granularity)
        num_basket = int(state['observation']['players'][0]['curr_basket'] + 1)
        num_items = int(len(list(selected_items)))
        num_checkout = int(len(list(purchased_items)))

        has_items, has_basket, has_checkout = 0,0,0

        if num_basket >= 1:
            if verbose:
                print("Has a basket!")
            has_basket = 1

            if num_items == 0:
                if verbose:
                    print("Has all items!")
                has_items = 1

            if num_checkout == 0:
                if verbose:
                    print("Purchased all items!")
                has_checkout = 1

        #encoding: ((((x,y)*2 + cart)*2 + items)*2 + checkout)
        idx = ((((player_x*self.height + player_y)*2 + has_basket)*2 + has_items)*2 + has_checkout)
        print(f"Has Basket: {has_basket}, Has Items: {has_items}, Has Checked Out: {has_checkout}")

        if return_checks:
            return idx, [has_basket, has_items, has_checkout]
        else:
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

        self.epsilon = max(0.0, self.epsilon - 0.005)

        return action