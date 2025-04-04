import game as model 
import game_config
import random

board = model.Board()
print("Initial Board:")
print(board)
input('Press enter to continue...')

print('Moving left...')
board.move(model.LEFT)
print(board)
input('Press enter to continue...')

print('Moving up...')
board.move(model.UP)
print(board)
input('Press enter to continue...')

print('Moving right...')
board.move(model.RIGHT)
print(board)
input('Press enter to continue...')

print('Moving down...')
board.move(model.DOWN)
print(board)
input('Press enter to continue...')

print('Playing game randomly until lost...')
while not board.check_loss():
    board.move(random.choice(model.DIRECTIONS))

print("Game Over!")
print(board)
print(f'Final score: {board.score}')