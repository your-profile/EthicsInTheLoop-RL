'''
Performs saved demonstrations based on PID.
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

'''
Reads saved demonstrations from pickle file.

Input: Filename
Output: Saved Demo Dictionary
'''
def read_demos(demo_filename=None):

    file = open('./data/{}.pickle'.format(demo_filename), 'rb')
    demo_dict = pickle.load(file)
    file.close()

    return demo_dict

if __name__ == "__main__":
    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']

    # Initialize Q-learning agent
    action_space = len(action_commands) - 1 
    agent = QLAgent(action_space)
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 1972
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    # Demo PID
    pid = input("Input the Participant ID: ")
    pid = f"{pid}_Demonstration"
    
    episode_length = 1000

    demonstration_dict = read_demos(pid)

    for i in demonstration_dict.keys():
        print(i)
        sock_game.send(str.encode("0 RESET"))  # reset the game
        state = recv_socket_data(sock_game)
        state = json.loads(state)
        cnt = 0

        episode_dict = demonstration_dict[0]
        print(episode_dict["actions"])

        for step in range(demonstration_dict[i]["steps"]):
            cnt += 1
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
            
            state = next_state

            if cnt > episode_length:
                break

        print(cnt)

    sock_game.close()

