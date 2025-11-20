from imitation.algorithms.mce_irl import MCEIRL
import pickle
from collections import defaultdict
import numpy as np

# Import pickle file of choice here
with open("data/1_Demonstration.pickle", "rb") as f:
    demonstrations = pickle.load(f)

transitionCounts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

def trans(state, granularity=1.0):
    # You should design a function to transform the huge state into a learnable state for the agent
    # It should be simple but also contains enough information for the agent to learn
    roundVal = int(1/granularity)

    playerX = round(state["observation"]["players"][0]["position"][0] * roundVal) / roundVal
    playerY = round(state["observation"]["players"][0]["position"][1] * roundVal) / roundVal

    hasBasket = int(len(state['observation']['baskets']))
    
    hasItem = state['observation']['players'][0]['holding_food']

    paid = state['observation']['players'][0]['bought_holding_food']

    return playerX, playerY, hasBasket, hasItem, paid

# convert the object to a MDP
# for each demonstration

# print(len(demonstrations))

for demonstrationNum in demonstrations:
    # print(demonstrations[demonstrationNum].keys())
    # for each state
    for i in range(len(demonstrations[demonstrationNum]["states"])-1):
        state = trans(demonstrations[demonstrationNum]["states"][i], 0.5)
        action = demonstrations[0]["actions"][i]
        nextState = trans(demonstrations[demonstrationNum]["states"][i+1], 0.5)

        transitionCounts[state][action][nextState] += 1

# transitionProbabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

# for state in transitionCounts:
#     for action in transitionCounts[state]:
#         count = len(transitionCounts[state][action])
#         for nextState in transitionCounts[state][action]:
#             transitionProbabilities[state][action][nextState] = transitionCounts[state][action][nextState] / count

# print(transitionProbabilities)

transitionProbabilities = {}

for state in transitionCounts:
    for action in transitionCounts[state]:
        count = len(transitionCounts[state][action])
        for nextState in transitionCounts[state][action]:
            transitionProbabilities[state,action,nextState] = transitionCounts[state][action][nextState] / count

# print(transitionProbabilities)