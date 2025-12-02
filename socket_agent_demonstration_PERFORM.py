#Author Hang Yu

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
exit_pos = [-0.8, 15.6] # The position of the exit in the environment from [-0.8, 15.6] in x, and y = 15.6
cart_pos_left = [1, 18.5] # The position of the cart in the environment from [1, 2] in x, and y = 18.5
cart_pos_right = [2, 18.5] 

def distance_to_cart(state):
    agent_position = state['observation']['players'][0]['position']
    if agent_position[0] > 1.5:
        cart_distances = [euclidean_distance(agent_position, cart_pos_right)]
    else:
        cart_distances = [euclidean_distance(agent_position, cart_pos_left)]
    return min(cart_distances)

def euclidean_distance(pos1, pos2):
    # Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5


def calculate_reward(previous_state, current_state):
    return 0


def read_demos(demo_filename=None):

    file = open('./data/{}.pickle'.format(demo_filename), 'rb')
    demo_dict = pickle.load(file)
    file.close()

    return demo_dict

if __name__ == "__main__":
    

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    agent = QLAgent(action_space)

####################
    #Once you have your agent trained, or you want to continue training from a previous training session, you can load the qtable from a json file
    #agent.qtable = pd.read_json('qtable.json')
####################
    
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 1972
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))
    pid = input("Input the Participant ID: ")
    pid = f"{pid}_Demonstration"
    
    training_time = 2
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

            reward = calculate_reward(state, next_state)  # You need to define this function
            
            agent.learning(action_index, reward, agent.trans(state), agent.trans(next_state))
            state = next_state

            agent.qtable.to_json('qtable.json')

            if cnt > episode_length:
                break

        print(cnt)

    sock_game.close()

