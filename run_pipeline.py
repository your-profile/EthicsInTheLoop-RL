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
import csv
from datetime import datetime

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
    
def initialize_csv_files():
    """Initialize CSV files with headers"""
    # Demo priming metrics
    with open('demo_priming_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['demo_name', 'total_episodes', 'total_steps', 'avg_steps_per_episode', 
                        'priming_violations', 'violations_per_step', 'qtable_states_populated', 'timestamp'])
    
    # Evaluation results 
    with open('evaluation_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['demo_name', 'eval_run', 'episode_num', 'success', 'steps_taken', 
                        'violations', 'has_basket_step', 'has_items_step', 'has_checkout_step', 
                        'final_position_x', 'final_position_y', 'timestamp'])
    
    # Summary statistics
    with open('summary_statistics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['demo_name', 'success_rate', 'avg_steps_success', 'avg_steps_all', 
                        'total_violations_eval', 'violation_rate_eval', 'violation_rate_demo', 
                        'improvement_ratio', 'timestamp'])

def log_priming_metrics(demo_name, total_episodes, total_steps, violations_count, qtable_size):
    """Log metrics from the priming phase"""
    with open('demo_priming_metrics.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        avg_steps = total_steps / total_episodes if total_episodes > 0 else 0
        violation_rate = violations_count / total_steps if total_steps > 0 else 0
        writer.writerow([demo_name, total_episodes, total_steps, avg_steps, 
                        violations_count, violation_rate, qtable_size, datetime.now()])

def log_evaluation_episode(demo_name, eval_run, episode_num, success, steps_taken, 
                          violations, has_basket_step, has_items_step, has_checkout_step,
                          final_pos_x, final_pos_y):
    """Log individual episode results during evaluation"""
    with open('evaluation_results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([demo_name, eval_run, episode_num, success, steps_taken, 
                        violations, has_basket_step, has_items_step, has_checkout_step,
                        final_pos_x, final_pos_y, datetime.now()])

def log_summary_statistics(demo_name, success_rate, avg_steps_success, avg_steps_all,
                          total_violations, violation_rate_eval, violation_rate_demo):
    """Log summary statistics for a demo"""
    with open('summary_statistics.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        improvement = ((violation_rate_demo - violation_rate_eval) / violation_rate_demo * 100) if violation_rate_demo > 0 else 0
        writer.writerow([demo_name, success_rate, avg_steps_success, avg_steps_all,
                        total_violations, violation_rate_eval, violation_rate_demo,
                        improvement, datetime.now()])

def prime_from_demos(sock_game):
    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    action_space = len(action_commands) - 1
    
    data_dir = 'data'
    p = Path(data_dir)
    pids = [entry.name for entry in p.iterdir() if entry.is_file()]
    print(f"All Demos: {pids}")
    
    for pid in pids:
        agent = QLAgent(action_space, epsilon=0.01)
        
        print(f"Now priming with: {pid}")
        demonstration_dict = read_demos(pid)

        # Track metrics for this demo
        total_episodes = len(demonstration_dict.keys())
        total_steps = 0
        priming_violations = 0  # Track violations during priming

        ## Q-PRIMING
        for episode_key in demonstration_dict.keys():
            print(f"Episode: {episode_key}")
            sock_game.send(str.encode("0 RESET"))
            state = recv_socket_data(sock_game)
            state = json.loads(state)
            cnt = 0

            episode_dict = demonstration_dict[episode_key]
            print(f"Actions in episode: {episode_dict['actions']}")

            for step in range(episode_dict["steps"]):
                cnt += 1
                total_steps += 1
                
                action_index = episode_dict["actions"][step]
                action = "0 " + action_commands[action_index]
                sock_game.send(str.encode(action))
                next_state = recv_socket_data(sock_game)
                
                if (((state['observation']['players'][0]['direction']) != (action_index - 1) and (action_index != 5))): 
                    sock_game.send(str.encode(action))
                    next_state = recv_socket_data(sock_game)
                
                if len(next_state) == 0 or state['observation']['players'][0]['position'][0] < 0.3:
                    break 

                next_state = json.loads(next_state)

                # Check for violations (if they're in the state)
                if 'violations' in next_state and next_state['violations'] != '':
                    priming_violations += len(next_state['violations']) if isinstance(next_state['violations'], list) else 1

                priming_value = 30
                agent.priming(action_index, priming_value, agent.trans(state), agent.trans(next_state))
                state = next_state

                if cnt >= episode_dict["steps"] - 1:
                    break
            
            print(f"Completed {cnt} steps")

        # Log priming metrics
        qtable_populated = len(agent.qtable[agent.qtable.sum(axis=1) > 0])
        pid_without_extension = pid.split(".")[0]
        log_priming_metrics(pid_without_extension, total_episodes, total_steps, 
                          priming_violations, qtable_populated)

        # Save qtable
        path_to_jsons = "pipeline_primed_qtables_json"
        path_to_pkls = "pipeline_primed_qtables_pkl"
        os.makedirs(path_to_jsons, exist_ok=True)
        os.makedirs(path_to_pkls, exist_ok=True)
        agent.qtable.to_json(f'{path_to_jsons}/{pid_without_extension}_pipeline_primed_qtable.json')
        save_qtable(agent, f'{path_to_pkls}/{pid_without_extension}_pipeline_primed_qtable.pkl')
        print(f"Saved qtable as {pid_without_extension}_pipeline_primed_qtable.json and {pid_without_extension}_pipeline_primed_qtable.pkl")       


# PERFORMING FROM PRIMED QTABLES / EVALUATION

def evaluate_primed_qtables_from_demos(sock_game):
    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    action_space = len(action_commands) - 1
    
    json_dir = 'pipeline_primed_qtables_json'
    p = Path(json_dir)
    qtables = [entry.name for entry in p.iterdir() if entry.is_file()]
    print(f"Evaluating the following primed qtables: {qtables}")
    
    # For debugging - comment out to run all
    # qtables = ['10G_Demonstration_pipeline_primed_qtable.json']
    
    for qt in qtables:
        agent = QLAgent(action_space)
        json_path = f"./{json_dir}/{qt}"
        agent.qtable = pd.read_json(json_path)
        
        demo_name = qt.replace('_pipeline_primed_qtable.json', '')
        print(f"Now evaluating: {demo_name}")
        
        # Metrics tracking
        num_eval_runs = 100
        episode_length = 1000
        successes = []
        steps_list = []
        total_violations = 0
        
        for eval_run in range(num_eval_runs):
            sock_game.send(str.encode("0 RESET"))
            state = recv_socket_data(sock_game)
            state = json.loads(state)
            cnt = 0
            
            # Track milestones
            has_basket_step = -1
            has_items_step = -1
            has_checkout_step = -1
            episode_violations = 0
            
            while not state['gameOver']:
                cnt += 1
                
                shopping_list = set(state['observation']['players'][0]['shopping_list'])
                selected_items = []
                purchased_items = []

                if len(state['observation']['baskets']) > 0:
                    basket_list = set(state['observation']['baskets'][0]['contents'])
                    purchased_list = set(state['observation']['baskets'][0]['purchased_contents'])
                    selected_items = shopping_list.difference(basket_list)
                    purchased_items = shopping_list.difference(purchased_list)

                has_basket = int(state['observation']['players'][0]['curr_basket'] + 1)
                has_items = len(state['observation']['baskets'][0]['contents']) if len(state['observation']['baskets']) > 0 else 0
                has_checkout = len(state['observation']['baskets'][0]['purchased_contents']) if len(state['observation']['baskets']) > 0 else 0

                if has_basket >= 1 and has_basket_step == -1:
                    has_basket_step = cnt
                if has_items > 0 and has_items_step == -1:
                    has_items_step = cnt
                if has_checkout > 0 and has_checkout_step == -1:
                    has_checkout_step = cnt
                    
                # Choose action
                action_index = agent.choose_action(agent.trans(state))
                action = "0 " + action_commands[action_index]

                sock_game.send(str.encode(action))
                next_state = recv_socket_data(sock_game)
                
                if (((state['observation']['players'][0]['direction']) != (action_index - 1) and (action_index != 5))): 
                    sock_game.send(str.encode(action))
                    next_state = recv_socket_data(sock_game)
                
                if len(next_state) == 0 or state['observation']['players'][0]['position'][0] < 0.3:
                    break 

                next_state = json.loads(next_state)
                
                # Count violations
                if 'violations' in next_state and next_state['violations'] != '':
                    violation_count = len(next_state['violations']) if isinstance(next_state['violations'], list) else 1
                    episode_violations += violation_count
                    total_violations += violation_count

                state = next_state

                if cnt >= episode_length:
                    break
            
            # Determine success
            success = (has_basket_step != -1 and has_items_step != -1 and has_checkout_step != -1)
            successes.append(1 if success else 0)
            steps_list.append(cnt)
            
            final_pos = state['observation']['players'][0]['position']
            
            # Log this episode
            log_evaluation_episode(demo_name, 1, eval_run, success, cnt, episode_violations,
                                 has_basket_step, has_items_step, has_checkout_step,
                                 final_pos[0], final_pos[1])
            
            if eval_run % 10 == 0:
                print(f"  Completed {eval_run}/{num_eval_runs} episodes")
        
        # Calculate summary statistics
        success_rate = sum(successes) / len(successes) * 100
        avg_steps_all = sum(steps_list) / len(steps_list)
        successful_steps = [steps_list[i] for i in range(len(successes)) if successes[i] == 1]
        avg_steps_success = sum(successful_steps) / len(successful_steps) if successful_steps else 0
        violation_rate_eval = total_violations / sum(steps_list) if sum(steps_list) > 0 else 0
        
        # Get demo violation rate from priming metrics
        violation_rate_demo = 0
        try:
            with open('demo_priming_metrics.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['demo_name'] == demo_name:
                        violation_rate_demo = float(row['violations_per_step'])
                        break
        except:
            pass
        
        # Log summary
        log_summary_statistics(demo_name, success_rate, avg_steps_success, avg_steps_all,
                             total_violations, violation_rate_eval, violation_rate_demo)
        
        print(f"\n=== {demo_name} Summary ===")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Avg Steps (All): {avg_steps_all:.1f}")
        print(f"Avg Steps (Success): {avg_steps_success:.1f}")
        print(f"Total Violations: {total_violations}")
        print(f"Violation Rate: {violation_rate_eval:.4f}")
        print("=" * 40 + "\n")      
    
# # Connect to Supermarket
# HOST = '127.0.0.1'
# PORT = 1972
# sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock_game.connect((HOST, PORT))
# # execute    
# prime_from_demos(sock_game)
# evaluate_primed_qtables_from_demos(sock_game)
# sock_game.close()
# # TODO get metrics and plot nicely in some way
        
# Connect to Supermarket
HOST = '127.0.0.1'
PORT = 1972
sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_game.connect((HOST, PORT))

# Initialize CSV files
initialize_csv_files()

# Execute pipeline
prime_from_demos(sock_game)
evaluate_primed_qtables_from_demos(sock_game)
sock_game.close()

print("Check the CSV files for results:")
print("  - demo_priming_metrics.csv")
print("  - evaluation_results.csv")
print("  - summary_statistics.csv")
        


