import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import json
import random
import socket
import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data
from Q_learning_agent_prime import QLAgent  # Make sure to import your QLAgent class
import pickle
import pandas as pd

granularity = 4.0
HEIGHT = int(26*granularity)
WIDTH = int(26*granularity)

# Policy network (GAIL generator)
class Generator(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,32), nn.ReLU(),
            nn.Linear(32,32), nn.ReLU(),
            nn.Linear(32,act_dim),
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
            nn.Linear(obs_dim+act_dim,32), nn.ReLU(),
            nn.Linear(32,32), nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid(),)

    def forward(self, obs, act_onehot):
        x = torch.cat([obs, act_onehot], dim=-1)
        return self.net(x)

# makes actions into one hot vector
def one_hot(a, act_dim):
    y = np.zeros(act_dim)
    y[a] = 1
    return y

# Function that takes a proper shopper states and transforms it into a state feature vector
def state_to_obs(state):

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
    # print(state['observation']['players'][0]['position'])
    player_x = round(state['observation']['players'][0]['position'][0] * 4.0)
    player_y = round(state['observation']['players'][0]['position'][1] * 4.0)
    num_basket = int(state['observation']['players'][0]['curr_basket'] + 1)
    num_items = int(len(list(selected_items)))
    num_checkout = int(len(list(purchased_items)))
    player_direction = state['observation']['players'][0]['direction']

    has_items, has_basket, has_checkout = 0, 0, 0

    if num_basket >= 1:
        has_basket = 1

        if num_items == 0:
            has_items = 1

        if num_checkout == 0:
            has_checkout = 1

    # encoding: ((((x,y)*2 + cart)*2 + items)*2 + checkout)
    feature_vector = np.array([player_x, player_y, has_basket, has_items, has_checkout])

    return feature_vector

#takes demonstration dict and separates feature states and actions
'''
Toy example with three timesteps

feature = [x, y, has_item, has_cart, has_chekcout]
expert_obs = [[x, y, 0, 1, 0] timestep 1 where (no items, has a cart, not checked out)
                [x, y, 0, 0, 1] timestep 2
                [x, y, 1, 0, 0] timestep 3
                ]

expert_actions = [0,6,3]
'''
def split_demonstrations(dictionary):

    expert_obs, expert_actions = [], []

    for i in range (len(dictionary)):
        states = dictionary[i]['states']
        for state in states[1:]:
            state_array = state_to_obs(state)
            expert_obs.append(state_array)
        for action in dictionary[i]['actions']:
            expert_actions.append(action)

    print(len(expert_obs), len(expert_actions))

    try:
        assert(len(expert_obs) == len(expert_actions))
    except:
        AssertionError("Misaligned State and Actions")
    
    return expert_obs, expert_actions

# Checks if the observation matches the current checkpoint and returns true if it does. Otherwise returns false
def is_episode_over(obs, checkpoint):
    
    if np.array_equal(obs, checkpoint):
        return True
    else:
        return False


# looks at primed q table and pull the most visited/highest value states. Return as state checkpoints
def calcuate_checkpoints_from_primedQ(table, n=5):

    collapsed = np.zeros((HEIGHT, WIDTH))
    vector = [0, 0, 0, 0, 0, 0, 0]
    top_checkpoints = []

    # add empty checkpoints
    for i in range(n):
        top_checkpoints.append(vector)

    # compute top5 cells
    for x in range(HEIGHT):
        for y in range(WIDTH):
            cell_value = 0
            for cart in [0, 1]:
                for items in [0, 1]:
                    for checkout in [0, 1]:
                        idx = ((((x * HEIGHT + y) * 2 + cart) * 2 + items) * 2 + checkout)

                        cell_value += table.loc[idx].sum()


                        vector = [idx, x, y, cart, items, checkout, cell_value]

                        # replace empty or lower checkpoints
                        for i, item in enumerate(top_checkpoints):
                            if cell_value > top_checkpoints[i][-1]:
                                top_checkpoints[i] = vector
                                break

            collapsed[x, y] = cell_value

    return top_checkpoints


# policy rollout trajectory collection
def collect_policy_trajectories(sock_game, policy, checkpoint, steps=800):
    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    
    obs_list, action_list = [], []
    done = False

    sock_game.send(str.encode("0 RESET"))  # reset the game
    state = recv_socket_data(sock_game)
    state = json.loads(state)
    obs = state_to_obs(state)

    for _ in range(steps):
        action_index, _ = policy.choose_action(obs)
        action = "0 " + action_commands[action_index]


        obs_list.append(obs)
        action_list.append(action_index)

        sock_game.send(str.encode(action))  # send action to env
        next_state = recv_socket_data(sock_game)  # get observation from env
        obs = state_to_obs(state)
        done = is_episode_over(obs, checkpoint) # determines if an episode has ended from the obs vector

        if done:
            sock_game.send(str.encode("0 RESET"))  # reset the game
            state = recv_socket_data(sock_game)
            state = json.loads(state)
            obs = state_to_obs(state)

    return np.array(obs_list), np.array(action_list)

# traininf loop
def train_gail(env, expert_obs, expert_actions, checkpoint, index, iterations=10, save_dir="gail_output", lr = 3e-4, ):
    #TODO: hard code obs space size and action space size
    obs_dim = 5
    act_dim = 7

    #generator and disctriminator
    policy = Generator(obs_dim, act_dim)
    discr = Discriminator(obs_dim, act_dim)

    opt_policy = optim.Adam(policy.parameters(), lr=lr)
    opt_discr = optim.Adam(discr.parameters(), lr=lr)

    expert_obs_t = torch.tensor(expert_obs, dtype=torch.float32)
    expert_act_onehot = torch.tensor([one_hot(a, act_dim) for a in expert_actions], dtype=torch.float32)

    # create directory for results
    os.makedirs(save_dir, exist_ok=True)

    for it in range(iterations):
        # policy rollouts
        policy_obs, policy_actions = collect_policy_trajectories(env, policy, checkpoint)
        policy_obs_tensor = torch.tensor(policy_obs, dtype=torch.float32)
        policy_action_onehot = torch.tensor([one_hot(a, act_dim) for a in policy_actions], dtype=torch.float32)
        policy_action_tensor = torch.tensor(policy_actions, dtype=torch.long)

        # trainig discriminator
        discriminator_expert = discr(expert_obs_t, expert_act_onehot)
        discriminator_policy = discr(policy_obs_tensor, policy_action_onehot)

        loss_expert = torch.mean(torch.log(discriminator_expert + 1e-8))
        loss_policy = torch.mean(torch.log(1 - discriminator_policy + 1e-8))
        discr_loss = -(loss_expert + loss_policy)
        opt_discr.zero_grad()
        discr_loss.backward()
        opt_discr.step()

        # rewards for policy
        with torch.no_grad():
            rewards = -torch.log(1 - discr(policy_obs_tensor, policy_action_onehot) + 1e-8)

        log_probs = []
        # update policy (PPO)

        for i in range(len(policy_obs)):
            logits = policy(torch.tensor(policy_obs[i], dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            log_probs.append(dist.log_prob(policy_action_tensor[i]))


        log_probs = torch.stack(log_probs)

        policy_loss = -(log_probs * rewards.squeeze()).mean() #loss
        opt_policy.zero_grad()
        policy_loss.backward()
        opt_policy.step()

        if it % 50 == 0:
            print(f"Iteration {it} | Discriminator Loss: {discr_loss.item():.3f} | Generator Loss: {policy_loss.item():.3f}")

    # saving outputs
    torch.save(policy.state_dict(), os.path.join(save_dir, f"GAIL_generator_{index}.pth"))
    torch.save(discr.state_dict(), os.path.join(save_dir, f"GAIL_discriminator_{index}.pth"))
    

    return policy, discr

# read demo files
def read_demos(demo_filename=None):

    file = open('./data/{}.pickle'.format(demo_filename), 'rb')
    demo_dict = pickle.load(file)
    file.close()

    return demo_dict

# main
def main():
    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    
    with open('primed_qtable.pkl', 'rb') as file:
        primed_qtable = pickle.load(file)

    checkpoints = calcuate_checkpoints_from_primedQ(primed_qtable)

    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 1972
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    '''
    TODO: Identify states from Q priming that are checkpoints. is_episode_over should recieve this checkpoint in feature vector form
    TODO: Then do IRL for each checkpoint and save them
    TODO: Use these reward functions for basline Q learning, but make sure to switch reward functions at each checkpoint
    '''

    pid = input("Input the Participant ID: ")
    pid = f"{pid}_Demonstration"

    demonstration_dict = read_demos(pid)

    #TODO: create a feature vector representation of our states
    # Then separate these observations and actions

    expert_obs, expert_actions = split_demonstrations(demonstration_dict)

    np.save(os.path.join("outputGail", f"expert_obs.npy"), expert_obs, allow_pickle=True)
    np.save(os.path.join("outputGail", f"expert_actions.npy"), expert_actions, allow_pickle=True)

    for i, checkpoint in enumerate(checkpoints):
        policy, discr = train_gail(sock_game, expert_obs, expert_actions, iterations=800, save_dir=f"outputGail_{i}", checkpoint=checkpoint, index=i)

if __name__ == "__main__":
    main()
