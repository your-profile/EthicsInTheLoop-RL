'''
After running GAIL, this file shows the agent using the N policies trained with the GAIl algorithm.
This file runs through each checkpoint. Make sure your checkpoint parameters are the same as in the GAIL training algorithm.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import json
import socket
from utils import recv_socket_data
import pickle
import itertools
import math
import argparse
import socket
import pickle
import os
import numpy as np

granularity = 4.0
HEIGHT = int(26*granularity)
WIDTH = int(26*granularity)

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

def find_checkpoint_for_state(state, checkpoints, radius=15):
    sx, sy, sb1, sb2, sb3 = state

    best_idx = None
    best_dist = float('inf')

    for i, cp in enumerate(checkpoints):
        _, cx, cy, cb1, cb2, cb3, _ = cp

        # binary features must match exactly
        if (sb1, sb2, sb3) != (cb1, cb2, cb3):
            continue

        # compute Euclidean distance
        dist = math.sqrt((sx - cx)**2 + (sy - cy)**2)

        # must be inside radius
        if dist <= radius:
            # keep the closest checkpoint
            if dist < best_dist:
                best_dist = dist
                best_idx = i

    return best_idx


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
def split_demonstrations(dictionary, checkpoints, radius = 30):

    expert = {}

    for key in range(len(checkpoints)):
        expert[key]= {}
        expert[key]["obs"] = []
        expert[key]["actions"] = []


    for i in range(len(dictionary)):
        states = dictionary[i]['states']
        for (state, action) in zip(states[1:], dictionary[i]['actions']):
            state_array = state_to_obs(state)
            key = find_checkpoint_for_state(state_array, checkpoints, radius=radius)
            # print(key, state_array)
            expert[key]['obs'].append(state_array)
            expert[key]['actions'].append(action)
    
    for key in expert.keys():
        assert(len(expert[key]["obs"]) == len(expert[key]["actions"]))

    
    # print(expert)
    
    return expert

# Checks if the observation matches the current checkpoint and returns true if it does. Otherwise returns false
def is_episode_over(state):
    
    if state['observation']['players'][0]['position'][0] < 0.3:
        return True
    else:
        return False
    

def calculate_checkpoints_from_primeQ(table, n=6, dist=5):
    vector = [0, 0, 0, 0, 0, 0, 0]
    top_checkpoints = []

    # add empty checkpoints
    for i in range(n):
        top_checkpoints.append(vector)

    binary_states = list(itertools.product([1,0], repeat=3))

    for cart, items, checkout in binary_states:
        for x in range(HEIGHT):
            for y in range(WIDTH):

                idx = ((((x*HEIGHT + y)*2 + cart)*2 + items)*2 + checkout)
                qvals = table.loc[idx].values
                cell_value = qvals.sum()
                vector = [idx, x, y, cart, items, checkout, cell_value]

                replaced = False
                near_any = False

                for i, item in enumerate(top_checkpoints):
                    _, cx, cy, _, _, _, old_value = item 
                    if (cx - dist) < x < (cx + dist) and (cy - dist) < y < (cy + dist):
                        near_any = True
                        if cell_value > old_value:
                            # print(f"{top_checkpoints[i]} | {vector}")
                            top_checkpoints[i] = vector
                            replaced = True
                        break

                if not replaced and not near_any:
                    lowest_index = min(range(len(top_checkpoints)),
                                    key=lambda j: top_checkpoints[j][-1])

                    if cell_value > top_checkpoints[lowest_index][-1]:
                        top_checkpoints[lowest_index] = vector

    return top_checkpoints


# policy rollout trajectory collection
def collect_policy_trajectories(sock_game, policies, checkpoints, steps=800):
    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    
    models = {0: Generator(5,7),
              1: Generator(5,7),
              2: Generator(5,7),
              3: Generator(5,7),
              4: Generator(5,7),
              5: Generator(5,7),
              6: Generator(5,7),
              7: Generator(5,7)}
    
    done = False

    sock_game.send(str.encode("0 RESET"))  # reset the game
    state = recv_socket_data(sock_game)
    state = json.loads(state)
    obs = state_to_obs(state)
    last_key = 0

    for i in range(steps):
        key = find_checkpoint_for_state(obs, checkpoints)
        model = models[key]

        model.load_state_dict(torch.load(f"/Users/juliasantaniello/Desktop/EthicsInTheLoop-RL/outputGail/GAIL_generator_{key}.pth", map_location="cpu"))
        model.eval()

        try:
            action_index, _ = model.choose_action(obs)
        except:
            key = last_key
            action_index, _ = model.choose_action(obs)


        action = "0 " + action_commands[action_index]

        sock_game.send(str.encode(action))  # send action to env
        next_state = recv_socket_data(sock_game)  # get observation from env
        state = json.loads(next_state)

        obs = state_to_obs(state)
        done = is_episode_over(state) # determines if an episode has ended from the obs vector


        if done or i > steps - 1:
            sock_game.send(str.encode("0 RESET"))  # reset the game
            state = recv_socket_data(sock_game)
            state = json.loads(state)
            obs = state_to_obs(state)

        last_key = key

# traininf loop
def train_gail(env, expert_dict, checkpoints, iterations=200, save_dir="outputGail", disc_lr = 3e-4, gen_lr = 3e-4):
    #TODO: hard code obs space size and action space size
    obs_dim = 5
    act_dim = 7

    #generator and disctriminator
    policies = {}
    expert_obs = {}
    expert_actions = {}


    for key in expert_dict.keys():
        policies[key] = {}
        expert_obs[key] = {}
        expert_actions[key] = {}

        policies[key]["policy"] = Generator(obs_dim, act_dim)
        policies[key]["disc"] = Discriminator(obs_dim, act_dim)

        policies[key]["opt_policy"] = optim.Adam(policies[key]["policy"].parameters(), lr=gen_lr)
        policies[key]["opt_discr"] = optim.Adam(policies[key]["disc"].parameters(), lr=disc_lr)

        expert_obs_array_x = expert_dict[key]['obs']
        expert_obs_array_x = np.array(expert_obs_array_x, dtype=np.float32)
        expert_obs_t_x = torch.from_numpy(expert_obs_array_x)
        # expert_obs_t_x = torch.tensor(expert_obs_array_x, dtype=torch.float32)

        expert_obs[key]['obs'] = expert_obs_t_x

        expert_actions_x = expert_dict[key]['actions']
        expert_act_onehot_array_x = np.array([one_hot(a, act_dim) for a in expert_actions_x])
        expert_act_onehot_x = torch.tensor(expert_act_onehot_array_x, dtype=torch.float32)

        expert_obs[key]['actions'] = expert_act_onehot_x


    # create directory for results
    os.makedirs(save_dir, exist_ok=True)

    for it in range(iterations):
        policy_obs, policy_actions = collect_policy_trajectories(env, policies, checkpoints)

    return policies

# read demo files
def read_demos(demo_filename=None):

    file = open('./data/{}.pickle'.format(demo_filename), 'rb')
    demo_dict = pickle.load(file)
    file.close()

    return demo_dict

# ordering of the binary groups
group_order = {
    (0,0,0): 0,
    (0,1,1): 0,
    (0,0,1): 0,
    (0,1,0): 0,
    (1,0,0): 1,
    (1,1,1): 1,
    (1,1,0): 2,
    (1,0,1): 3,
}

def sort_key(entry):
    idx = entry[0]
    group = (entry[3], entry[4], entry[5])
    group_index = group_order[group]

    if group == (0,0,0):
        # ascending vector
        return (group_index, idx)
    else:
        # descending vector
        return (group_index, -idx)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--radius", type=int, default=40,
                        help="radius for matching demos to checkpoints")
    parser.add_argument("--dist", type=int, default=6,
                        help="distance threshold for calculating checkpoints")
    parser.add_argument("--n", type=int, default=5,
                        help="number of checkpoints to extract from primed Q-table")
    parser.add_argument("--qtable", type=str, default="primed_qtable_20G.pkl",
                        help="path to primed qtable file")
    parser.add_argument("--iterations", type=int, default=800,
                        help="number of GAIL training iterations")

    return parser.parse_args()


def main():
    args = parse_args()

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    action_space = len(action_commands) - 1

    with open(args.qtable, 'rb') as f:
        primed_qtable = pickle.load(f)

    checkpoints = calculate_checkpoints_from_primeQ(
        primed_qtable,
        n=args.n,
        dist=args.dist
    )

    HOST = '127.0.0.1'
    PORT = 1972
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    pid = input("Input the Participant ID: ")
    pid = f"{pid}_Demonstration"

    demonstration_dict = read_demos(pid)

    sorted_checkpoints = sorted(checkpoints, key=sort_key)
    print(sorted_checkpoints)

    expert_dict = split_demonstrations(
        demonstration_dict,
        sorted_checkpoints,
        radius=args.radius
    )

    policies = train_gail(
        sock_game,
        expert_dict,
        iterations=args.iterations,
        save_dir="outputGail",
        checkpoints=sorted_checkpoints
    )


if __name__ == "__main__":
    main()