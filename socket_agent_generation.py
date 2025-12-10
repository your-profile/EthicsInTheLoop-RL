#Author: Hang Yu

import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from Q_learning_agent_prime import QLAgent  # Make sure to import your QLAgent class
import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from socket_agent_training_GAIL import state_to_obs

# Policy network (GAIL generator)
class Generator(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU(),
            nn.Linear(64,act_dim),
        )

    def forward(self, obs):
        logits = self.net(obs)
        return logits

    def choose_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        output = self.forward(obs)
        dist = torch.distributions.Categorical(logits=output)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# discriminating network (determines difference between expert and policy rollout)
class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim+act_dim,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid(),)

    def forward(self, obs, act_onehot):
        x = torch.cat([obs, act_onehot], dim=-1)
        return self.net(x)

def collect_policy_trajectories(sock_game, policy, steps=800):
        action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
        
        sock_game.send(str.encode("0 RESET"))  # reset the game
        state = recv_socket_data(sock_game)
        state = json.loads(state)

        for i in range(steps):
            obs = state_to_obs(state)


            action_index, _ = policy.choose_action(obs)
            action = "0 " + action_commands[action_index]

            sock_game.send(str.encode(action))  # send action to env
            next_state = recv_socket_data(sock_game)  # get observation from env
            state = json.loads(next_state)
            obs = state_to_obs(state)




            if i > steps - 1:
                sock_game.send(str.encode("0 RESET"))  # reset the game
                state = recv_socket_data(sock_game)
                state = json.loads(state)
                obs = state_to_obs(state)




if __name__ == "__main__":
    

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    
    HOST = '127.0.0.1'
    PORT = 1972
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    model = Generator(5,7)
    model.load_state_dict(torch.load("/Users/juliasantaniello/Desktop/EthicsInTheLoop-RL/outputGail/GAIL_generator_3.pth", map_location="cpu"))
    model.eval()

    collect_policy_trajectories(sock_game=sock_game, policy=model)


    # training_time = 100
    # episode_length = 1000
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

    #         # Update state
    #         state = next_state

    #         if cnt >= episode_length:
    #             break

    # Close socket connection
    sock_game.close()

