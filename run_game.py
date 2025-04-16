import sys
import pygame
import multiprocessing as mp
import game_files.game as model
from expectimax import getNextBestMove
from math import log2
from multiprocessing import Manager, Pool
from game_files.game_config import BOARD_SIZE, DIRECTIONS
import random
import csv
import matplotlib.pyplot as plt
import pandas as pd 

depth = 2
FPS = 244
size = width, height = 780, 800
playRegion = 780, 780

black = (0, 0, 0)
white = (255, 255, 255)
fontColor = (82, 52, 42)
defaultTileColor = (211, 211, 211)

COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

def create_pool():
    return mp.Pool(processes=4)

pygame.init()
screen = pygame.display.set_mode(size)
pygame.display.set_caption("2048")
tileFont = pygame.font.Font("./fonts/ClearSans-Regular.ttf", 40)
scoreFont = pygame.font.SysFont("./fonts/ClearSans-Regular.ttf", 20)

#draw the board for the 2048 game
def drawBoard(screen, board, agent):
    #clear screen
    screen.fill(black)
    #tile size
    tile_width = playRegion[0] / BOARD_SIZE
    tile_height = playRegion[1] / BOARD_SIZE
    
    #draw tiles
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            color = defaultTileColor
            numberText = ''
            tile = board.board[i][j]
            
            #if tile exists then assign a color
            if tile is not None: 
                tile_value = tile.value
                if tile_value in COLORS.keys():
                    color = COLORS.get(tile_value)
                else:
                    color = (60, 58, 50)

                numberText = str(tile_value)
            
            #get position of tiles 
            x_pos = j * tile_width
            y_pos = i * tile_height
            rect = pygame.Rect(x_pos, y_pos, tile_width, tile_height)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, fontColor, rect, 1)

            #set tile text and font
            fontImage = tileFont.render(numberText, True, fontColor)
            if fontImage.get_width() > tile_width:
                new_width = tile_width
                new_height = fontImage.get_height() / fontImage.get_width() * tile_width
                fontImage = pygame.transform.scale(fontImage,(new_width, new_height))
                
            #center text
            text_x = x_pos + (tile_width - fontImage.get_width())/2
            text_y = y_pos + (tile_height - fontImage.get_height())/2
            screen.blit(fontImage, (text_x, text_y))
    score_text = (f"Score: {board.score}")
    if agent is not None:
        score_text += (f"[agent enabled, depth = {depth}]")
    
    #display score and render game
    fontImage = scoreFont.render(score_text, True, white)
    screen.blit(fontImage, (1, playRegion[1] + 1))

#get keyboard input for game
def handleInput(event, board):
    global agent

    if event.type == pygame.QUIT:
        sys.exit()
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_RIGHT:
            board.move(model.RIGHT)
        elif event.key == pygame.K_LEFT:
            board.move(model.LEFT)
        elif event.key == pygame.K_UP:
            board.move(model.UP)
        elif event.key == pygame.K_DOWN:
            board.move(model.DOWN)
        elif event.key == pygame.K_r:
            board = model.Board(BOARD_SIZE)
        elif event.key == pygame.K_ESCAPE:
            sys.exit()

    return board

#play episode manually
def play_episode():
    global depth
    clock = pygame.time.Clock()
    board = model.Board(BOARD_SIZE)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            board = handleInput(event, board)

        if board.checkLoss():
            print("Game Over")
            print(f"Final Score: {board.score}")
            pygame.quit()
            return

        drawBoard(screen, board, agent=None)
        pygame.display.flip()
        clock.tick(FPS)
        
#run visual expectimax episode
def expectimax_episode():
    global depth
    agent = True
    clock = pygame.time.Clock()
    board = model.Board(BOARD_SIZE)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return 0, 0
            board = handleInput(event, board)

        #check if game is over
        if board.checkLoss():
            max_tile = max(tile.value if tile else 0 for row in board.board for tile in row)
            print("Game Over")
            print(f"Final Score: {board.score}")
            pygame.quit()
            return board.score, max_tile
            
        if agent:
            nextBestMove = getNextBestMove(board, depth)
            board.move(nextBestMove)

        drawBoard(screen, board, agent)
        pygame.display.flip()
        clock.tick(FPS)

#run no gui mode for expectimax
def expectimax_episode_headless():
    board = model.Board(BOARD_SIZE)
    while not board.checkLoss():
        move = getNextBestMove(board, depth)
        board.move(move)
    max_tile = max(tile.value if tile else 0 for row in board.board for tile in row)
    return board.score, max_tile

#run no gui mode for random sampling
def random_episode_headless():
    board = model.Board(BOARD_SIZE)
    while not board.checkLoss():
        move = random.choice(DIRECTIONS)
        board.move(move)
    max_tile = max(tile.value if tile else 0 for row in board.board for tile in row)
    return board.score, max_tile

#run batch random sampling
def run_random_batch(num_episodes=1000, csv_file='random_scores.csv'):
    scores = []

    for episode in range(num_episodes):
        score, max_tile = random_episode_headless()
        print(f"Episode {episode + 1}/{num_episodes} - Score: {score}, Max Tile: {max_tile}")
        scores.append((episode + 1, score, max_tile))

    # Write to CSV
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Score', 'Max Tile'])
        writer.writerows(scores)

    print(f"\nFinished {num_episodes} episodes. Scores saved to {csv_file}.")

#run batch expectimax 
def run_expectimax_batch(num_episodes=1000, csv_file='expectimax_scores.csv'):
    scores = []

    for episode in range(num_episodes):
        score, max_tile = expectimax_episode_headless()
        print(f"Episode {episode + 1}/{num_episodes} - Score: {score} | Max Tile: {max_tile}")
        scores.append((episode + 1, score, max_tile))

    # Write to CSV
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Score', 'Max Tile'])
        writer.writerows(scores)

    print(f"\nFinished {num_episodes} episodes. Scores saved to {csv_file}.")

#visualize random sample game
def random_sample():
    global depth
    agent = True
    clock = pygame.time.Clock()
    board = model.Board(BOARD_SIZE)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if board.checkLoss():
            print("Game Over")
            print(f"Final Score: {board.score}")
            pygame.quit()
            return

        random_direction = random.choice(DIRECTIONS)
        
        moved = board.move(random_direction)

        if moved:
            print(f"Move: {random_direction} | Score: {board.score}")
        
        drawBoard(screen, board, agent)
        pygame.display.flip()
        clock.tick(FPS)
        
#### PLOTTING DATA ####

base_path = "./scores/"
expectimax_file = base_path + "expectimax_scores.csv"
random_file = base_path + "random_scores.csv"
dqn_scores_file = base_path + "DQN_scores.csv"
dqn_tiles_file = base_path + "DQN_max_tiles.csv"

expectimax_df = pd.read_csv(expectimax_file)
random_df = pd.read_csv(random_file)
dqn_scores_df = pd.read_csv(dqn_scores_file)
dqn_tiles_df = pd.read_csv(dqn_tiles_file)


dqn_df = dqn_scores_df.merge(dqn_tiles_df, on='Episode')

        
def plot_max_tile_frequencies():
    plt.figure(figsize=(12, 6))

    def get_freq(df, label):
        return df['Max Tile'].value_counts().sort_index(), label

    exp_freq, exp_label = get_freq(expectimax_df, 'Expectimax')
    rnd_freq, rnd_label = get_freq(random_df, 'Random')

    all_tiles = sorted(set(exp_freq.index).union(rnd_freq.index))

    x = range(len(all_tiles))
    bar_width = 0.25

    plt.bar([i - bar_width for i in x], [exp_freq.get(tile, 0) for tile in all_tiles],
            width=bar_width, label=exp_label)
    plt.bar(x, [rnd_freq.get(tile, 0) for tile in all_tiles],
            width=bar_width, label=rnd_label)

    plt.xticks(ticks=x, labels=all_tiles)
    plt.xlabel('Max Tile')
    plt.ylabel('Frequency')
    plt.title('Max Tile Frequency Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Score Over Episodes Plot
def plot_scores_over_episodes():
    plt.figure(figsize=(14, 6))

    plt.plot(expectimax_df['Episode'], expectimax_df['Score'], label='Expectimax', alpha=0.7)
    plt.plot(random_df['Episode'], random_df['Score'], label='Random', alpha=0.7)

    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Score per Episode for Each Agent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_DQN_tiles():
    plt.figure(figsize=(12, 6))

    def get_freq(df, label):
        return df['Max Tile'].value_counts().sort_index(), label

    dqn_freq, dqn_label = get_freq(dqn_df, 'DQN')

    all_tiles = sorted(dqn_freq.index)

    x = range(len(all_tiles))
    bar_width = 0.25

    plt.bar(x, [dqn_freq.get(tile, 0) for tile in all_tiles], width=bar_width, label=dqn_label, align='center')

    plt.xticks(ticks=x, labels=all_tiles)
    plt.xlabel('Max Tile')
    plt.ylabel('Frequency')
    plt.title('Max Tile Frequency Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_DQN_scores():
    plt.figure(figsize=(14, 6))

    plt.plot(dqn_df['Episode'], dqn_df['Score'], label='DQN')

    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Score per Episode for Each Agent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
        


if __name__ == '__main__':
    #run_expectimax_batch()
    expectimax_episode()
    #random_sample()
    #run_random_batch()
    #play_episode()
    #plot_max_tile_frequencies()
    #plot_scores_over_episodes()
    #plot_DQN_scores()
    #plot_DQN_tiles()