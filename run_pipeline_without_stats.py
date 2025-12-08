'''
file goals:
# for loop through demos, get save qtables, eval q tables with performing, get norm rate and success rate 
# future work - add stochasticity to compare results because right now it is deterministic

done:
# modified socket_agent_training.py code to run on all demos and save jsons and pkls
 - dont need second half after input("Enter to go to Training: ") - commented out because we maybe want to add back later
# modified socket_agent_performing.py code to run on all demos 

TODO
# get norms broken in demo
# get norms broken after priming 
# get success rate 
# plots

RUNNING INSTRUCTIONS:
python3 socket_env.py --headless
python ./run_pipeline.py > angela_testing_file.txt
'''

import json
import random
import socket
import os

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from Q_learning_agent_prime import QLAgent  # Make sure to import your QLAgent class
import pickle
import pandas as pd
from pathlib import Path

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


def calculate_reward(previous_state, state):
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
    has_basket = int(state['observation']['players'][0]['curr_basket'] + 1)
    has_items = int(len(list(selected_items)))
    has_checkout = int(len(list(purchased_items)))

    if has_basket >= 1 and has_items == 0 and has_checkout == 0 and state['observation']['players'][0]['position'][0] < 0.3:
        return 100

    if state['observation']['players'][0]['position'][0] < 0.3:
        return -10
    
    return -1

def read_demos(demo_filename=None):

    file = open('./data/{}'.format(demo_filename), 'rb')
    demo_dict = pickle.load(file)
    file.close()

    return demo_dict

def save_qtable(agent, filename="qtable.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(agent.qtable, f)
    # print(f"Q-table saved to {filename}")

def prime_from_demos(sock_game):
    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    
    
    # loop through each demo in data directory
    data_dir = 'data'
    p = Path(data_dir)
    pids = [entry.name for entry in p.iterdir() if entry.is_file()]
    print(f"All Demos: {pids}")
    
    # GO THROUGH EACH DEMO AND PRIME Q TABLE
    for pid in pids:
        agent = QLAgent(action_space, epsilon=0.01)
        
        print(f"Now priming with: {pid}")
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

                priming_value = 30
                
                agent.priming(action_index, priming_value, agent.trans(state), agent.trans(next_state))
                state = next_state


                if cnt >= demonstration_dict[i]["steps"] - 1:
                    break
            
            print(cnt)

        # save qtable as pkl and json
        pid_without_extension = pid.split(".")[0]
        path_to_jsons = "pipeline_primed_qtables_json"
        path_to_pkls = "pipeline_primed_qtables_pkl"
        os.makedirs(path_to_jsons, exist_ok=True)
        os.makedirs(path_to_pkls, exist_ok=True)
        agent.qtable.to_json(f'{path_to_jsons}/{pid_without_extension}_pipeline_primed_qtable.json')
        save_qtable(agent, f'{path_to_pkls}/{pid_without_extension}_pipeline_primed_qtable.pkl')
        print(f"Saved qtable as {pid_without_extension}_pipeline_primed_qtable.json and {pid_without_extension}_pipeline_primed_qtable.pkl")

        # UNCOMMENT & MODIFY IF WE WANT TO DO FURTHER TRAINING :)!
        # input("Enter to go to Training: ")

        # for i in range(training_time):
        #     sock_game.send(str.encode("0 RESET"))  # reset the game
        #     state = recv_socket_data(sock_game)
        #     state = json.loads(state)
        #     cnt = 0
        #     while not state['gameOver']:
        #         cnt += 1
        #         # Choose an action based on the current state
        #         action_index = agent.choose_action(agent.trans(state))
        #         action = "0 " + action_commands[action_index]

        #         sock_game.send(str.encode(action))  # send action to env
        #         next_state = recv_socket_data(sock_game)  # get observation from env
                
        #         if (((state['observation']['players'][0]['direction']) != (action_index - 1) and (action_index != 5))): 
        #             sock_game.send(str.encode(action))  # send action to env
        #             next_state = recv_socket_data(sock_game)  # get observation from env
                
        #         if len(next_state) == 0 or state['observation']['players'][0]['position'][0] < 0.3:
        #             break 

        #         next_state = json.loads(next_state)

        #         # Define the reward based on the state and next_state
        #         reward = calculate_reward(state, next_state)  # You need to define this function

        #         if state['observation']['players'][0]['position'][0] < 0.4:
        #             print("------------------------------------------")
        #             print(reward, action_commands[action_index])
        #             print("------------------------------------------")
        #         # Update Q-table
        #         agent.learning(action_index, reward, agent.trans(state), agent.trans(next_state))

        #         # Update state
        #         state = next_state
        #         agent.qtable.to_json('qtable_2.json')

        #         if cnt >= episode_length:
        #             break
        #     # Additional code for end of episode if needed

        # Close socket connection
        # sock_game.close()
        
# PERFORMING FROM PRIMED QTABLES / EVALUATION

def evaluate_primed_qtables_from_demos(sock_game):
    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    
    # loop through each demo in data directory
    json_dir = 'pipeline_primed_qtables_json'
    p = Path(json_dir)
    qtables = [entry.name for entry in p.iterdir() if entry.is_file()]
    print(f"Evaluating the following primed qtables: {qtables}")
    
    # for debugging:
    qtables = ['10G_Demonstration_pipeline_primed_qtable.json']
    
    for qt in qtables:
        agent = QLAgent(action_space)
        json_path = f"./{json_dir}/{qt}"
        agent.qtable = pd.read_json(json_path)
        print(agent.qtable)
        print(f"Now evaluating: {qt}")
        # input()
        
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

                # Define the reward based on the state and next_state
                # reward = calculate_reward(state, next_state)  # You need to define this function

                # if state['observation']['players'][0]['position'][0] < 0.4:
                #     print("------------------------------------------")
                #     print(reward, action_commands[action_index])
                #     print("------------------------------------------")
                # Update Q-table
                # agent.learning(action_index, reward, agent.trans(state), agent.trans(next_state))

                # Update state
                state = next_state
                # agent.qtable.to_json('qtable_2.json')

                if cnt >= episode_length:
                    break
            # Additional code for end of episode if needed

        # Close socket connection
        # sock_game.close()
    
# Connect to Supermarket
HOST = '127.0.0.1'
PORT = 1972
sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_game.connect((HOST, PORT))
# execute    
prime_from_demos(sock_game)
evaluate_primed_qtables_from_demos(sock_game)
sock_game.close()
# TODO get metrics and plot nicely in some way
        
        


