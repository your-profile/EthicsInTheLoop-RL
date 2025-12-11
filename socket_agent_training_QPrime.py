'''
This file primes Q tables using the demonstrations saved.
'''

import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from Q_learning_agent_prime import QLAgent  # Make sure to import your QLAgent class
import pickle
import pandas as pd


cart = False

def calculate_reward(previous_state, state):
    #shopping list
    shopping_list = set(state['observation']['players'][0]['shopping_list'])
    selected_items = []
    purchased_items = []

    # if you have a basket, calculate how many items you have that are purchased and unpurchased
    if len(state['observation']['baskets']) > 0:
        basket_list = set(state['observation']['baskets'][0]['contents'])
        purchased_list = set(state['observation']['baskets'][0]['purchased_contents'])
        selected_items = shopping_list.difference(basket_list)
        purchased_items = shopping_list.difference(purchased_list)

    #calculate features
    has_basket = int(state['observation']['players'][0]['curr_basket'] + 1)
    has_items = int(len(list(selected_items)))
    has_checkout = int(len(list(purchased_items)))

    if has_basket >= 1 and has_items == 0 and has_checkout == 0 and state['observation']['players'][0]['position'][0] < 0.3:
        return 100 #winning

    if state['observation']['players'][0]['position'][0] < 0.3:
        return -10 #loss
    
    return -1 #step

def read_demos(demo_filename=None):

    file = open('./data/{}.pickle'.format(demo_filename), 'rb')
    demo_dict = pickle.load(file)
    file.close()

    return demo_dict

def save_qtable(agent, filename="primed_qtable.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(agent.qtable, f)
    print(f"Q-table saved to {filename}")

if __name__ == "__main__":
    

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    agent = QLAgent(action_space, epsilon=0.01)

    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 1972
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))
    pid = input("Input the Participant ID: ")
    pid = f"{pid}_Demonstration"
    training_time = 100
    episode_length = 1000

    demonstration_dict = read_demos(pid)

    ## Q-PRIMING

    for i in demonstration_dict.keys():
        print(i)
        sock_game.send(str.encode("0 RESET"))  # reset the game
        state = recv_socket_data(sock_game)
        state = json.loads(state)
        cnt = 0

        episode_dict = demonstration_dict[i]
        print(episode_dict["actions"])

        for step in range(demonstration_dict[i]["steps"]):
            cnt += 1
            # print(step, demonstration_dict[i]["steps"])
            action_index = demonstration_dict[i]["actions"][step]
            action = "0 " + action_commands[action_index]
            sock_game.send(str.encode(action))  # send action to env
            next_state = recv_socket_data(sock_game)  # get observation from env
            
            if (((state['observation']['players'][0]['direction']) != (action_index - 1) and (action_index != 5))): 
                sock_game.send(str.encode(action))  # send action to env
                next_state = recv_socket_data(sock_game)  # get observation from env
            
            if len(next_state) == 0 or state['observation']['players'][0]['position'][0] < 0.3:
                break 

            next_state = json.loads(next_state)

            priming_value = 40
            
            agent.priming(action_index, priming_value, agent.trans(state), agent.trans(next_state))
            state = next_state


            if cnt >= demonstration_dict[i]["steps"] - 1:
                break
        
        print(cnt)

    agent.qtable.to_json('primed_qtable.json')
    save_qtable(agent)

    # Close socket connection
    sock_game.close()

