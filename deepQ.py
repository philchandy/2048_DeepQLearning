import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np
from collections import namedtuple, deque
from itertools import count
from game_files.game import Board 
import csv
from expectimax import snakeHeuristic, max_tile_in_corner, edge_bonus, maximize_empty_cells
import copy
import os
from game_files.game_config import BOARD_SIZE, LEFT, RIGHT, UP, DOWN



# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# State Encoder
def encode_state(board):
    #converts 2048 board to tensor
    grid = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            #get each tiles value in board
            tile = board.board[i][j]
            grid[i][j] = 0 if tile is None else tile.value
    #flatten board to 1D, convert value to log base 2 (values now 0-16 (max value for 2048))
    board_flat = [0 if val == 0 else int(math.log(val, 2)) for val in grid.flatten()]
    board_tensor = torch.LongTensor(board_flat)
    #convert to one hot vectors based on int representation of logbase2(value)
    board_tensor = F.one_hot(board_tensor, num_classes=16).float().flatten()
    #board reshape after one hot, 1 sample, 4x4 grid, 16 channels(for each tile val)
    board_tensor = board_tensor.reshape(1, 4, 4, 16).permute(0, 3, 1, 2)
    return board_tensor

# DQN
#save single step (agent experience)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

#class to store and manage past experiences
class ReplayMemory:
    def __init__(self, capacity):
        #initialize deque, throw out oldest memory when capacity reached
        self.memory = deque([], maxlen=capacity)

    #save new step
    def push(self, *args):
        self.memory.append(Transition(*args))

    #randomly sample batch of memories to train with
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#neural net
class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=2),
            nn.SiLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=2),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(16, 128)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# params
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 20
n_actions = 4

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
memory = ReplayMemory(50000)
steps_done = 0

#epsilon greedy action selection
def select_action(state):
    global steps_done
    sample = random.random()
    #eps decay
    eps_threshold = max(EPS_END, EPS_START * (EPS_DECAY ** steps_done))
    steps_done += 1
    
    #get valid actions in specific state
    valid = valid_actions(game)
    if sample > eps_threshold:
        #exploit
        with torch.no_grad():
            #forward pass through policy net to get predicted q-values for actions
            q_values = policy_net(state)
            #get only valid actions
            valid_q = [(i, q_values[0][i].item()) for i in valid]
            #choose action with highest q value 
            best_action = max(valid_q, key=lambda x: x[1])[0]
            return torch.tensor([[best_action]], device=device, dtype=torch.long)
    else:
        #explore
        return torch.tensor([[random.choice(valid)]], device=device, dtype=torch.long)


def optimize_model():
    #update weights 
    if len(memory) < BATCH_SIZE:
        #dont update if there isnt enough data in mem
        return
    #sample batch of experiences
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    #create mask to track transitions that are non-terrminal 
    non_final_mask_values = []
    for state in batch.next_state:
        non_final_mask_values.append(state is not None)

    non_final_mask = torch.tensor(non_final_mask_values, device=device, dtype=torch.bool)
    
    valid_next_states = []
    for state in batch.next_state:
        if state is not None:
            valid_next_states.append(state)

    non_final_next_states = torch.cat(valid_next_states)
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #predict q values from curr policy net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    if non_final_next_states.size(0) > 0:
        # Double DQN:
        next_q_values = policy_net(non_final_next_states)
        #use policy net to get the best action in next state
        next_actions = next_q_values.max(1)[1].unsqueeze(1)
        next_state_q_values = target_net(non_final_next_states).gather(1, next_actions).squeeze()
        #use target net to estimate q value for the action
        next_state_values[non_final_mask] = next_state_q_values.detach()

    #bellman
    expected_values = reward_batch + GAMMA * next_state_values

    loss = nn.MSELoss()(state_action_values, expected_values.unsqueeze(1))
    optimizer.zero_grad()
    #backprop
    loss.backward()
    #ugradient descent, update policy net weights
    optimizer.step()

def same_move(state, next_state, last_memory):
    #Check if any of the tensors are None
    if state is None or next_state is None or last_memory.state is None or last_memory.next_state is None:
        return False

    same_state = torch.equal(state, last_memory.state)
    same_next_state = torch.equal(next_state, last_memory.next_state)
    
    return same_state and same_next_state

def moving_average(data, window_size=10):
    #caluclate moving avg for debugging 
    if len(data) < window_size:
        return None
    return sum(data[-window_size:]) / window_size

def valid_actions(board):
    valid = []
    for i in range(4):
        new_board = copy.deepcopy(board)
        if new_board.move(action_map[i]):
            valid.append(i)
    return valid

def get_last_episode(filename):
    if not os.path.exists(filename):
        return -1  # No file, start from 0

    with open(filename, 'r') as file:
        lines = list(csv.reader(file))
        if len(lines) <= 1:
            return -1 
        try:
            last_row = lines[-1]
            return int(last_row[0])
        except:
            return -1

def save_scores(scores, filename, header):
    last_ep = get_last_episode(filename)
    start_ep = last_ep + 1

    file_exists = os.path.exists(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists or last_ep == -1:
            writer.writerow(header)
        latest_score = scores[-1]
        writer.writerow([start_ep, latest_score])


# load models
policy_net_path = './networks/policy_net2.pth'
target_net_path = './networks/target_net2.pth'

try:
    policy_net.load_state_dict(torch.load(policy_net_path))
    target_net.load_state_dict(torch.load(target_net_path))
    print("Loaded previous models.")
except FileNotFoundError:
    print("No previous models found. Starting training from scratch.")

# Training
num_episodes = 3000
total_scores = []
best_tiles = []

max_avg_score = float('-inf')
#early_stop_counter = 0
#early_stop_patience = 10

action_map = {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}

#batch training
for i_episode in range(num_episodes):
    
    print(f"\n=== Episode {i_episode + 1}/{num_episodes} ===")
    game = Board()
    state = encode_state(game).float()
    last_score = 0

    for t in count():
        #print(f"\nStep {t} ")
        action = select_action(state)
        #print(f"Selected Action: {action.item()} ({['UP','DOWN','LEFT','RIGHT'][action.item()]})")

        move_success = game.move(action_map[action.item()])
        #print(f"Move Success: {move_success}")

        #reward_val = (game.score - last_score) + 0.1 * game.get_empty_positions()
        reward_val = reward_val = (
            #use heuristic to shape rewards
            (game.score - last_score)
            + 10 * maximize_empty_cells(game)
            + 4 * snakeHeuristic(game)
            + 3 * max_tile_in_corner(game)
            + 2 * edge_bonus(game)
        )
        reward = torch.tensor([reward_val], device=device, dtype=torch.float)
        #print(f"Reward before penalty: {reward_val}")

        last_score = game.score
        done = game.checkLoss()

        next_state = encode_state(game).float() if not done else None

        if next_state is not None and torch.equal(state, next_state):
            reward -= 20
            #print("Penalty applied for no change in state.")

        if next_state is None or len(memory) == 0 or not same_move(state, next_state, memory.memory[-1]):
            memory.push(state, action, next_state, reward)

        state = next_state
        optimize_model()

        if done:
            print("\n>>> Game Over <<<")
            max_tile = max(tile.value if tile else 0 for row in game.board for tile in row)
            print(f"Final Score: {game.score}")
            print(f"Max Tile: {max_tile}")
            total_scores.append(game.score)
            best_tiles.append(max_tile)
            
            save_scores(total_scores, 'episode_scores2.csv', ['Episode', 'Score'])
            save_scores(best_tiles, 'max_tiles2.csv', ['Episode', 'Max Tile'])

            if i_episode >= 50:
                avg = sum(total_scores[-50:]) / 50
                print(f"Last 50 Episodes Avg Score: {avg:.2f}")
            break

    #update target net
    if i_episode % TARGET_UPDATE == 0:
        print("Updating target network.")
        #copy weights from policy net to target net
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode % 100 == 0:
        print("Saving models.")
        torch.save(policy_net.state_dict(), 'policy_net2.pth')
        torch.save(target_net.state_dict(), 'target_net2.pth')

print("\nTraining Complete")
print(f"Total Episodes Run: {num_episodes}")
print(f"Average Score: {sum(total_scores) / len(total_scores):.2f}")
print(f"Max Tile Achieved: {max(best_tiles)}")
