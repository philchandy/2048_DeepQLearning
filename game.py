import random
import random
import multiprocessing as mp
import expectimax
from game_config import BOARD_SIZE, DIRECTIONS, LEFT, RIGHT, UP, DOWN

import random

class Tile:
    def __init__(self, value):
        self.value = value
        self.merged = False

class Board:
    def __init__(self, board_size=BOARD_SIZE):
        self.boardSize = board_size
        self.board = [[None for _ in range(self.boardSize)] for _ in range(self.boardSize)]
        self.score = 0
        self.add_tile()
        self.add_tile()

    def add_tile(self):
        empty_positions = self.get_empty_positions()
        if not empty_positions:
            return
        row, col = random.choice(empty_positions)
        rand_tile = 0
        if random.random() < 0.9:
            rand_tile = 2
        else:
            rand_tile = 4
        self.board[row][col] = Tile(rand_tile)

    def get_empty_positions(self):
        empty_pos = []
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.board[i][j] is None:
                    empty_pos.append((i,j))
        return empty_pos

    def reset_merge(self):
        for row in self.board:
            for tile in row:
                if tile:
                    tile.merged = False

    def move(self, direction):
        self.reset_merge()
        moved = False

        if direction in [LEFT, RIGHT]:
            for r in range(self.boardSize):
                row = self.board[r] if direction == LEFT else self.board[r][::-1]
                if self.slide_and_merge(row):
                    moved = True
                if direction == RIGHT:
                    self.board[r] = row[::-1]
                else:
                    self.board[r] = row

        elif direction in [UP, DOWN]:
            for c in range(self.boardSize):
                column = [self.board[r][c] for r in range(self.boardSize)]
                if direction == DOWN:
                    column.reverse()
                if self.slide_and_merge(column):
                    moved = True
                if direction == DOWN:
                    column.reverse()
                for r in range(self.boardSize):
                    self.board[r][c] = column[r]

        if moved:
            self.add_tile()
        return moved

    def slide_and_merge(self, line):
        new_line = []
        for tile in line:
            if tile:
                new_line.append(tile)
                
        moved = False
        if len(new_line) < len(line):
            moved = True
        
        i = 0
        while i < len(new_line) - 1:
            if new_line[i].value == new_line[i+1].value:
                if not new_line[i].merged and not new_line[i+1].merged:
                    new_line[i].value*= 2
                    self.score += new_line[i].value
                    new_line[i].merged = True
                    new_line.pop(i+1)
                    moved = True  
            i += 1
        
        while len(new_line) < self.boardSize:
            new_line.append(None)
        
        changed = False
        for i in range(self.boardSize):
            if line[i] != new_line[i]:
                changed = True
            line[i] = new_line[i]
            
        if changed:
            moved = True
        return moved

    def checkLoss(self):
        if self.get_empty_positions():
            return False
        for r in range(self.boardSize):
            for c in range(self.boardSize - 1):
                if self.board[r][c].value == self.board[r][c + 1].value:
                    return False
        for c in range(self.boardSize):
            for r in range(self.boardSize - 1):
                if self.board[r][c].value == self.board[r + 1][c].value:
                    return False
        return True
