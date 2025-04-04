import copy
import random
from game_config import BOARD_SIZE, DIRECTIONS


INF = 2**64
PERFECT_SNAKE = [[2,   2**2, 2**3, 2**4],
                 [2**8, 2**7, 2**6, 2**5],
                 [2**9, 2**10,2**11,2**12],
                 [2**16,2**15,2**14,2**13]]


def snakeHeuristic(board):
    #ideal values in perfect snake
    h = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            h += board.board[i][j].value * PERFECT_SNAKE[i][j] if board.board[i][j] else 0
    return h

def max_tile_in_corner(board):
    corners = [(0,0), (0, BOARD_SIZE-1), (BOARD_SIZE-1,0), (BOARD_SIZE-1, BOARD_SIZE-1)]
    max_tile = 0
    for i, j in corners:
        if board.board[i][j]:
            max_tile = max(max_tile, board.board[i][j].value)
    return max_tile

def edge_bonus(board):
    score = 0
    for i, row in enumerate(board.board):
        for j, tile in enumerate(row):
            if tile and (i in {0, BOARD_SIZE-1} or j in {0, BOARD_SIZE-1}):
                score += tile.value
    return score

def maximize_empty_cells(board):
    return len(board.get_empty_positions())

def merge_bonus(board):
    #add function in main to get when scores increased?
    return 0
    
def heuristic_function(board):
    sum = (4 *snakeHeuristic(board) 
           + (10 * maximize_empty_cells(board)) 
           + (3 * max_tile_in_corner(board))
           + (2 * edge_bonus(board)) 
           + (4 * merge_bonus(board)))
    return sum

def getNextBestMove(board, depth=2, isExpectimax=True):
    bestScore = -INF
    bestNextMove = DIRECTIONS[0]
    
    for dir in DIRECTIONS:
        copyBoard = copy.deepcopy(board)
        moved = copyBoard.move(dir) 
        if not moved:
            continue
        if isExpectimax==True:
            score, _ = expectimax(copyBoard, depth, dir)
        else: 
            score, _ = expectiminimax(copyBoard, depth, dir)
            
        if score >= bestScore:
            bestScore = score
            bestNextMove = dir

    return bestNextMove

def expectimax(board, depth, dir=None):
    from game import Tile
    
    if board.checkLoss():
        return -INF, dir
    elif depth == 0:
        return heuristic_function(board), dir

    if depth % 2 == 1:
        bestScore = -INF
        bestMove = None
        for dir in DIRECTIONS:
            copyBoard = copy.deepcopy(board)
            moved = copyBoard.move(dir)

            if moved:
                score = expectimax(copyBoard, depth - 1, dir)[0]
                if score > bestScore:
                    bestScore = score
                    bestMove = dir
        return bestScore, bestMove

    else:
        return 0, None

def expectiminimax(board, depth, dir=None):
    from game import Tile
    #return -inf for bad states
    if board.checkLoss():
        return -INF, dir
    elif depth == 0:
        return heuristic_function(board), dir

    #agent moves on odd
    if depth % 2 == 1:
        bestScore = -INF
        bestMove = None
        for dir in DIRECTIONS:
            copyBoard = copy.deepcopy(board)
            moved = copyBoard.move(dir)
            #if valid move call expectiminimax function to get score
            if moved:
                score = expectiminimax(copyBoard, depth - 1, dir)[0]
                if score > bestScore:
                    bestScore = score
                    bestMove = dir
        return bestScore, bestMove

    #random move (nature)
    else:
        bestScore = 0
        #get empoty tiles 
        openTiles = board.get_empty_positions()
        avgerage_score = 0
        for r, c in openTiles:
            #add tile to board and call expectiminimax on it
            #get avg score based on prob of tile being placed at r, c
            prob_empty = 1.0 / len(openTiles) #prob of a tile being placed in any of the empty spots
            copy_value_2 = copy.deepcopy(board)
            copy_value_4 = copy.deepcopy(board)
            
            copy_value_2.board[r][c] = Tile(2) 
            # 90% chance for 2 to spawn
            score_2 = 0.9 * expectiminimax(copy_value_2, depth - 1, dir)[0]
            
            copy_value_4.board[r][c] = Tile(4) 
            #10% chance for 4 to spawn
            score_4 = 0.1 * expectiminimax(copy_value_4, depth - 1, dir)[0]
            
            avgerage_score += ((score_2 + score_4) * prob_empty)

        return avgerage_score, dir