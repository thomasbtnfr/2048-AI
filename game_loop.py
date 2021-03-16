from Grid import Grid
from GameDriver import GameDriver
from minimax import getBestMove
import time

def startRemoteController():
    gameDriver = GameDriver()
    moves_str = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    moves_count = 1

    while True:
        grid = gameDriver.getGrid()
        #grid.printGrid()
        if grid.isGameOver():
            print("Unfortunately, I lost the game.")
            break
        if grid.nbEmpty() < 2:
            depth = 8
        elif grid.nbEmpty() < 4:
            depth = 6
        elif grid.nbEmpty() < 6:
            depth = 4
        else:
            depth = 3
        moveCode = getBestMove(grid, depth)
        print(f'Move #{moves_count}: {moves_str[moveCode[0]]} | Utility {moveCode[1]}, NbEmpty {grid.nbEmpty()}')
        gameDriver.move(moveCode[0])
        moves_count += 1

def startTerminal():
    moves_str = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    moves_count = 1

    grid = Grid([[]])
    grid.initializeGrid()
    grid.add2Or4()
    grid.add2Or4()

    while True:
        grid.printGrid()
        if grid.isGameOver():
            print("Unfortunately, I lost the game.")
            break
        depth = 5
        moveCode = getBestMove(grid, depth)
        print(f'Move #{moves_count}: {moves_str[moveCode[0]]} | Utility {moveCode[1]}, NbEmpty {grid.nbEmpty()}')
        grid.move(moveCode[0])
        grid.add2Or4()
        moves_count += 1

startTerminal()
#startRemoteController()

# [1024, 1024, 256, 1024, 2048, 2048, 1024, 1024, 1024, 1024]