from pynput import keyboard
import socket, json
from utils import recv_socket_data
from Q_Learning_agent import QLAgent

current_action = 0
action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT']

def on_press(key):
    global current_action
    try:
        if key == keyboard.Key.up:
            current_action = 1
        elif key == keyboard.Key.down:
            current_action = 2
        elif key == keyboard.Key.right:
            current_action = 3
        elif key == keyboard.Key.left:
            current_action = 4
        elif key.char == 'c':
            current_action = 5
        elif key == keyboard.Key.shift:
            current_action = 6
    except AttributeError:
        pass

def on_release(key):
    global current_action
    current_action = 0

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def choose_human_action():
    return current_action

# --- main loop ---
HOST, PORT = "127.0.0.1", 1972
sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_game.connect((HOST, PORT))
agent = QLAgent(len(action_commands) - 1)

while True:
    action_index = choose_human_action()
    if action_index != 0:  # only send if action pressed
        action = "0 " + action_commands[action_index]
        sock_game.send(str.encode(action))
        next_state = recv_socket_data(sock_game)
        next_state = json.loads(next_state)
        print("Action:", action_commands[action_index])
