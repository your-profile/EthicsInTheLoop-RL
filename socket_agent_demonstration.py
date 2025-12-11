'''
Saves expert dmeonstrations.
Must click on the second Black Window in order for the program to accept keyboard teleoperation.
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
import pygame

pygame.init()
red = (255, 0, 0)
green = (0, 255, 0)

'''
Simple Reward Function

Input: State, Next_State
Output: Reward
'''
def calculate_reward(previous_state, state):

    # Shopping List
    shopping_list = set(state['observation']['players'][0]['shopping_list'])

    #Unpurchased, selected items
    selected_items = []

    #purchsed items
    purchased_items = []

    # if you have a basket, calculate how many items you have that are purchased and unpurchased
    if len(state['observation']['baskets']) > 0:
        basket_list = set(state['observation']['baskets'][0]['contents'])
        purchased_list = set(state['observation']['baskets'][0]['purchased_contents'])
        selected_items = shopping_list.difference(basket_list)
        purchased_items = shopping_list.difference(purchased_list)


    # features for winning
    has_basket = int(state['observation']['players'][0]['curr_basket'] + 1)
    has_items = int(len(list(selected_items)))
    has_checkout = int(len(list(purchased_items)))

    if has_basket >= 1 and has_items == 0 and has_checkout == 0 and state['observation']['players'][0]['position'][0] < 0.3:
        return 100

    if state['observation']['players'][0]['position'][0] < 0.3:
        return -10
    
    return -1

'''
Allows for teleoperation of the propper shopper agent.
**MUST SELECT THE SECOND, BLANK WINDOW TO ALLOW KEYBOARD TO INFLUENCE AGENT**

Output: Human Action from Keyboard
'''
def choose_human_action():

    while True:
        #get event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        #get all pressed keys
        keys = pygame.key.get_pressed()

        # return action based on keyboard mapping
        if keys[pygame.K_UP]:
            return 1
        elif keys[pygame.K_LEFT]:
            return 4
        elif keys[pygame.K_DOWN]:
            return 2
        elif keys[pygame.K_RIGHT]:
            return 3
        elif keys[pygame.K_RETURN]:
            return 6
        elif keys[pygame.K_c]:
            return 5


'''
Saves demonstration
Input: Demo Dictionary and Filename
'''
def save_demo(demonstration_dict, demo_filename):
    with open('./data/{}.pickle'.format(demo_filename), 'wb') as handle:
        pickle.dump(demonstration_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved as:", demo_filename)


'''
Demonstration saved in dictionary.
States, actions, rewards, steps and other variables are saved at each step.
Saved per epsiode.

Output: Dictionary for an episode/trajectory
'''
def save_demonstration(environment_name, steps, states, timestamps, actions, rewards, seed, environment_version):
        print(len(rewards), (len(states) - 1), len(actions), final_episode_steps)
        
        assert(len(rewards) == (len(states) - 1) == len(actions))

        state_dict = {}

        state_dict["environment_name"] = environment_name
        state_dict["environment_version"] = environment_version
        state_dict["seed"] = seed
        state_dict["steps"] = steps
        state_dict["timestamps"] = timestamps
        state_dict["states"] = states
        state_dict["actions"] = actions
        state_dict["rewards"] = rewards

        return state_dict

if __name__ == "__main__":
    

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    
    # Q learning agent
    agent = QLAgent(action_space)
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 1972
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    # participant/demo ID
    pid = input("Input the Participant ID: ")
    pid = f"{pid}_Demonstration"
    
    #Pygame screen
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Supermarket Control Here")
    screen.fill(red)
    pygame.time.wait(5000)
    screen.fill(green)


    # training time and step length
    training_time = 20
    episode_length = 1000

    demonstration_dict = {}
    end = 0

    for i in range(training_time):
        final_episode_steps, final_episode_states, final_episode_rewards, final_episode_actions  = [], [], [], []

        sock_game.send(str.encode("0 RESET"))  # reset the game
        state = recv_socket_data(sock_game)
        state = json.loads(state)
        final_episode_states.append(state)

        cnt = 0

        while not state['gameOver']:
            cnt += 1
            pygame.time.wait(10)

            # Choose an action based on the current state
            action_index = choose_human_action()
            print("Chosen Action: ", action_index)
            action = "0 " + action_commands[action_index]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            print("Sending action: ", action)
            sock_game.send(str.encode(action))  # send action to env
            next_state = recv_socket_data(sock_game)  # get observation from env

            
            if (((state['observation']['players'][0]['direction']) != (action_index - 1)) and (action_index != 5)): 
                sock_game.send(str.encode(action))  # send action to env
                next_state = recv_socket_data(sock_game)  # get observation from env
            
            
            if len(next_state) == 0 or state['observation']['players'][0]['position'][0] < 0.3:
                break 
            
            next_state = json.loads(next_state)
        

            # Define the reward based on the state and next_state
            reward = calculate_reward(state, next_state)  # You need to define this function
            print("------------------------------------------")
            print(reward, action_commands[action_index])
            print("------------------------------------------")
            # Update Q-table
            agent.learning(action_index, reward, agent.trans(state), agent.trans(next_state))
            state = next_state

            # Update state
            final_episode_actions.append(action_index)
            final_episode_rewards.append(reward)
            final_episode_states.append(state)

            agent.qtable.to_json('qtable.json')

            if cnt > episode_length:
                break

        episode = save_demonstration(environment_name="properShopper", steps = cnt, states=final_episode_states, timestamps=None, actions=final_episode_actions, rewards=final_episode_rewards, seed=None, environment_version=None)
        demonstration_dict[i] = episode

    # Close socket connection
    save_demo(demonstration_dict, pid)
    print("Number ended: ", end)
    sock_game.close()

