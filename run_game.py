import sys
import pygame
import multiprocessing as mp
import game as model
from expectimax import getNextBestMove
from math import log2
from multiprocessing import Manager, Pool
from game_config import BOARD_SIZE, DIRECTIONS
import random

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

def drawBoard(screen, board, agent):
    screen.fill(black)
    tile_width = playRegion[0] / BOARD_SIZE
    tile_height = playRegion[1] / BOARD_SIZE
    
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            color = defaultTileColor
            numberText = ''
            tile = board.board[i][j]
            
            
            if tile is not None: 
                tile_value = tile.value
                if tile_value in COLORS.keys():
                    color = COLORS.get(tile_value)
                else:
                    color = (60, 58, 50)

                numberText = str(tile_value)
            
            x_pos = j * tile_width
            y_pos = i * tile_height
            rect = pygame.Rect(x_pos, y_pos, tile_width, tile_height)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, fontColor, rect, 1)

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
    fontImage = scoreFont.render(score_text, True, white)
    screen.blit(fontImage, (1, playRegion[1] + 1))

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
        
def expectimax_episode():
    global depth
    agent = True
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
            
        if agent:
            nextBestMove = getNextBestMove(board, depth, isExpectimax=True)
            board.move(nextBestMove)

        drawBoard(screen, board, agent)
        pygame.display.flip()
        clock.tick(FPS)

def expectiminimax_episode():
    global depth
    agent = True
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
            
        if agent:
            nextBestMove = getNextBestMove(board, depth, isExpectimax=False)
            board.move(nextBestMove)

        drawBoard(screen, board, agent)
        pygame.display.flip()
        clock.tick(FPS)

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
    

if __name__ == '__main__':
    expectiminimax_episode()
    #expectimax_episode()
    #random_sample()
    #play_episode()