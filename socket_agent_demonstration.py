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


pygame.init()
red = (255, 0, 0)
green = (0, 255, 0)

def choose_human_action():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        keys = pygame.key.get_pressed()

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


        # pygame.time.wait(10)

def save_demo(demonstration_dict, demo_filename):
    with open('./data/{}.pickle'.format(demo_filename), 'wb') as handle:
        pickle.dump(demonstration_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved as:", demo_filename)


'''
Demonstration saved in dictionary.
States, actions, rewards, steps and other variables are saved at each step.
Saved per epsiode.
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
    
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Supermarket Control Here")
    screen.fill(red)


    training_time = 20
    episode_length = 1000
    demonstration_dict = {}
    pygame.time.wait(5000)
    screen.fill(green)
    end = 0

    for i in range(training_time):
        final_episode_steps, final_episode_states, final_episode_rewards, final_episode_actions  = [], [], [], []

        sock_game.send(str.encode("0 RESET"))  # reset the game
        state = recv_socket_data(sock_game)
        print(state)
        state = json.loads(state)
        cnt = 0

        final_episode_states.append(state)


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


        # Additional code for end of episode if needed

    # Close socket connection
    save_demo(demonstration_dict, pid)
    print("Number ended: ", end)
    sock_game.close()

