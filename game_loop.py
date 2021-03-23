from Grid import Grid
from GameDriver import GameDriver
from minimax import getBestMove
import time
import sys, os
import numpy as np
import pandas as pd
from collections import Counter
import random

def startRemoteController():
    gameDriver = GameDriver()
    moves_str = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    moves_count = 1

    while True:
        grid = gameDriver.getGrid()
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

def startTerminalMinMax():
    moves_str = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    moves_count = 1

    grid = Grid([[]])
    grid.initializeGrid()

    while True:
        grid.printGrid()
        if grid.isGameOver():
            print("Unfortunately, I lost the game.")
            break
        depth = 5
        moveCode = getBestMove(grid, depth)
        grid.move(moveCode[0])
        grid.add2Or4()
        os.system('cls' if os.name=='nt' else 'clear')
        print(f'Move #{moves_count}: {moves_str[moveCode[0]]} | Utility {round(moveCode[1],2)} | Score {grid.score} | NbEmpty {grid.nbEmpty()}')
        moves_count += 1
    
    return grid.score, grid.maxValue()

def startTerminalRandom():
    moves_str = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    moves_count = 1


    grid = Grid([[]])
    grid.initializeGrid()

    while True:
        grid.printGrid()
        if grid.isGameOver():
            print("Unfortunately, I lost the game.")
            break
        moves = grid.getAvailableMovesForMax()
        move_chosen = random.choice(moves)
        grid.move(move_chosen)
        grid.add2Or4()
        os.system('cls' if os.name=='nt' else 'clear')
        print(f'Move #{moves_count}: {moves_str[move_chosen]} | Score {grid.score} | NbEmpty {grid.nbEmpty()}')
        moves_count += 1

    return grid.score, grid.maxValue()


def startTerminal(algoName: str):
    if algoName == 'minmax':
        return startTerminalMinMax()
    elif algoName == 'random':
        return startTerminalRandom()
    else:
        print('Unknown algorithm')



def writeResultat(fileName: str, maxTile: int, score: int):
    resultats = []
    resultats.append(str(maxTile) + ';' + str(score) + '\n')
    with open(fileName + '.txt', 'a') as f:
        f.write(''.join(resultats))


def evaluateModel(NbGame: int, modelName: str):
    for i in range(NbGame):
        score, maxTile = startTerminal(modelName)
        writeResultat(modelName, maxTile, score)
        

evaluateModel(10, 'random')
#startTerminal()
#startRemoteController()

"""
grid = Grid([[]])
matrix = [
    [256,64,0,0],
    [128,0,0,0],
    [0,0,256,0],
    [0,0,0,0],
]
grid.setMatrix(matrix)
print(grid.smoothness())
"""