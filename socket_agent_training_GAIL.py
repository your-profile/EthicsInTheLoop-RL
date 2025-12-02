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


# Policy network (GAIL generator)
class Generator(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
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
            nn.Linear(obs_dim + act_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, obs, act_onehot):
        x = torch.cat([obs, act_onehot], dim=-1)
        return self.net(x)

# makes actions into one hot vector
def one_hot(a, act_dim):
    y = np.zeros(act_dim)
    y[a] = 1
    return y

#TODO: Finish Function
def state_to_obs(state):
    x = None
    
    try:
        assert(len(x) > 0)
    except:
        AssertionError("Function not completed")

    return state

#takes demonstration dictionary and separates feature states and actions
def split_demonstrations(dictionary):

    expert_obs, expert_actions = [], []

    #expert_obs = 2d array of time x features
    #expert_actions = 1D array of actions

    # will need to transfer like so:
    #obs = state_to_obs(state)

    try:
        assert(len(expert_obs)>0)
    except:
        AssertionError("Function not written")
    
    return expert_obs, expert_actions

#TODO: Finish Function
def is_episode_over(obs):
    x = None
    
    try:
        assert(len(x) > 0)
    except:
        AssertionError("Episode Over Function not completed")

    return obs

# looks at primed q table and pull the most visited/highest value states. Return as state checkpoints
def calcuate_checkpoints_from_primedQ(table):
    checkpoints_states = []
    x = []

    try:
        assert(len(x) > 0)
    except:
        AssertionError("Function not completed")

    return checkpoints_states

def find_checkpoints(table):
    '''
    TODO: find the state chekcpoints from the primed Q table. 
    TODO: Then convert each to feature obs. Return the list of obs
    '''
    checkpoint_states = calcuate_checkpoints_from_primedQ(table)
    checkpoint_obs = []

    for c in checkpoint_states:
        c_obs = state_to_obs(c)
        checkpoint_obs.append(c_obs)

    return c_obs


# policy rollout trajectory collection
def collect_policy_trajectories(sock_game, policy, steps=1000):
    obs_list, action_list = [], []
    done = False

    sock_game.send(str.encode("0 RESET"))  # reset the game
    state = recv_socket_data(sock_game)
    state = json.loads(state)
    obs = state_to_obs(state) #TODO: finish state to obs one hot vector

    for _ in range(steps):
        action, _ = policy.choose_action(obs)

        obs_list.append(obs)
        action_list.append(action)

        sock_game.send(str.encode(action))  # send action to env
        next_state = recv_socket_data(sock_game)  # get observation from env
        obs = state_to_obs(state)
        done = is_episode_over(obs) #TODO: Function that determines if an episode has ended from the obs vector

        if done:
            sock_game.send(str.encode("0 RESET"))  # reset the game
            state = recv_socket_data(sock_game)
            state = json.loads(state)
            obs = state_to_obs(state) #TODO: finish state to obs one hot vector

    return np.array(obs_list), np.array(action_list)

# traininf loop
def train_gail(env, expert_obs, expert_actions, checkpoint, iterations=1000, save_dir="gail_output"):
    #TODO: hard code obs space size and action space size
    obs_dim = None #FIGURE OUT
    act_dim = 7

    #generator and disctriminator
    policy = Generator(obs_dim, act_dim)
    discr = Discriminator(obs_dim, act_dim)

    opt_policy = optim.Adam(policy.parameters(), lr=3e-4)
    opt_discr = optim.Adam(discr.parameters(), lr=3e-4)

    expert_obs_t = torch.tensor(expert_obs, dtype=torch.float32)
    expert_act_onehot = torch.tensor([one_hot(a, act_dim) for a in expert_actions], dtype=torch.float32)

    # create directory for results
    os.makedirs(save_dir, exist_ok=True)

    for it in range(iterations):
        # policy rollouts
        pol_obs, pol_actions = collect_policy_trajectories(env, policy)

        pol_obs_t = torch.tensor(pol_obs, dtype=torch.float32)
        pol_act_one = torch.tensor([one_hot(a, act_dim) for a in pol_actions], dtype=torch.float32)
        pol_act_t = torch.tensor(pol_actions, dtype=torch.long)

        # trainig discriminator
        D_exp = discr(expert_obs_t, expert_act_onehot)
        D_pol = discr(pol_obs_t, pol_act_one)

        discr_loss = -torch.mean(torch.log(D_exp + 1e-8) + torch.log(1 - D_pol + 1e-8))
        opt_discr.zero_grad()
        discr_loss.backward()
        opt_discr.step()

        # rewards for policy
        with torch.no_grad():
            rewards = -torch.log(1 - discr(pol_obs_t, pol_act_one) + 1e-8)

        # update policy (PPO)
        log_probs = []
        for i in range(len(pol_obs)):
            logits = policy(torch.tensor(pol_obs[i], dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            log_probs.append(dist.log_prob(pol_act_t[i]))
        log_probs = torch.stack(log_probs)

        policy_loss = -(log_probs * rewards.squeeze()).mean() #loss
        opt_policy.zero_grad()
        policy_loss.backward()
        opt_policy.step()

        if it % 50 == 0:
            print(f"Iteration {it} | Disc Loss: {discr_loss.item():.3f} | Policy Loss: {policy_loss.item():.3f}")

    # saving outputs
    torch.save(policy.state_dict(), os.path.join(save_dir, f"GAIL_generator_{checkpoint}.pth"))
    torch.save(discr.state_dict(), os.path.join(save_dir, f"GAIL_discriminator_{checkpoint}.pth"))
    

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
    
    with open('primed_qtable.json', 'r') as file:
        primed_qtable = json.load(file)

    checkpoints = find_checkpoints(primed_qtable) #TODO: Return major checkpoints as feature observations

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

    np.save(os.path.join("outputGail", f"expert_obs.npy"), expert_obs)
    np.save(os.path.join("outputGail", f"expert_actions.npy"), expert_actions)

    for i, checkpoint in enumerate(checkpoints):
        policy, discr = train_gail(sock_game, expert_obs, expert_actions, iterations=2000, save_dir=f"outputGail_{i}", checkpoint=i)

if __name__ == "__main__":
    main()
