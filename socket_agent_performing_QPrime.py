'''
This file shows the agent performing from the primed Q tables alone.
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


if __name__ == "__main__":
    

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    agent = QLAgent(action_space, epsilon=0.0)
    agent.qtable = pd.read_json('primed_qtable_20G.json')

    
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 1972
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    training_time = 100
    episode_length = 1000
    for i in range(training_time):
        sock_game.send(str.encode("0 RESET"))  # reset the game
        state = recv_socket_data(sock_game)
        state = json.loads(state)
        cnt = 0
        while not state['gameOver']:
            cnt += 1
            # Choose an action based on the current state
            action_index = agent.choose_action(agent.trans(state))
            action = "0 " + action_commands[action_index]

            sock_game.send(str.encode(action))  # send action to env
            next_state = recv_socket_data(sock_game)  # get observation from env
            
            if (((state['observation']['players'][0]['direction']) != (action_index - 1) and (action_index != 5))): 
                sock_game.send(str.encode(action))  # send action to env
                next_state = recv_socket_data(sock_game)  # get observation from env
            
            if len(next_state) == 0 or state['observation']['players'][0]['position'][0] < 0.3:
                break 

            next_state = json.loads(next_state)

            # Update state
            state = next_state

            if cnt >= episode_length:
                break

    # Close socket connection
    sock_game.close()

