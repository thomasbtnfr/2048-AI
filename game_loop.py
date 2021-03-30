from Grid import Grid
from GameDriver import GameDriver
from minimax import getBestMove
from expectiminimax import getBestMoveEMM
from SupervisedNN import flattenMatrix
from MCTS import mcts_move
import time
import sys, os
import numpy as np
import pandas as pd
from collections import Counter
import random
import tensorflow as tf


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

        tmp = [0,0,0,0]
        tmp[moveCode[0]] = 1
        matrixSupervised = grid.getMatrix()
        for i in range(4):
            for j in range(4):
                if matrixSupervised[i][j] == 0:
                    matrixSupervised[i][j] = 1
                matrixSupervised[i][j] = (np.log2(matrixSupervised[i][j]) / np.log2(65536))
        writeResultat('supervisedData',flattenMatrix(matrixSupervised), tmp)

        grid.move(moveCode[0])
        grid.add2Or4()
        os.system('cls' if os.name=='nt' else 'clear')
        print(f'Move #{moves_count}: {moves_str[moveCode[0]]} | Utility {round(moveCode[1],2)} | Score {grid.score} | NbEmpty {grid.nbEmpty()}')
        moves_count += 1
    
    return grid.score, grid.maxValue()

def startTerminalEmm():
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
        moveCode = getBestMoveEMM(grid, depth)
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

def startTerminalSupervised(model):
    moves_count = 1

    grid = Grid([[]])
    grid.initializeGrid()

    while True:
        grid.printGrid()
        if grid.isGameOver():
            print("Unfortunately, I lost the game.")
            break
        pred = model.predict(np.array([flattenMatrix(grid.getMatrix())]))
        pred = pred[0]
        orderedMoveCode = np.argsort(pred)[::-1]
        for move in orderedMoveCode:
            if (move == 0 and grid.canMoveUp()) or (move == 1 and grid.canMoveDown()) or (move == 2 and grid.canMoveLeft()) or (move == 3 and grid.canMoveRight()):
                grid.move(move)
                break
        grid.add2Or4()
        os.system('cls' if os.name=='nt' else 'clear')
        print(f'Move #{moves_count} | Score {grid.score}')
        moves_count += 1

    return grid.score, grid.maxValue()

def startTerminalMCTS():
    moves_str = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    moves_count = 1

    grid = Grid([[]])
    grid.initializeGrid()

    while True:
        grid.printGrid()
        if grid.isGameOver():
            print("Unfortunately, I lost the game.")
            break
        moveCode = mcts_move(grid, 40, 40)
        grid.move(moveCode)
        grid.add2Or4()
        os.system('cls' if os.name=='nt' else 'clear')
        print(f'Move #{moves_count}: {moves_str[moveCode]} | Score {grid.score} | NbEmpty {grid.nbEmpty()}')
        moves_count += 1
    
    return grid.score, grid.maxValue()

def writeResultat(fileName: str, maxTile: int, score: int):
    resultats = []
    resultats.append(str(maxTile) + ';' + str(score) + '\n')
    with open(fileName + '.txt', 'a') as f:
        f.write(''.join(resultats))

def startTerminal(algoName: str):
    if algoName == 'minmax':
        return startTerminalMinMax()
    elif algoName == 'random':
        return startTerminalRandom()
    elif algoName == "emm":
        return startTerminalEmm()
    elif algoName == "supervised":
        return startTerminalSupervised(tf.keras.models.load_model('supervisedModel'))
    elif algoName == 'mcts':
        return startTerminalMCTS()
    else:
        print('Unknown algorithm')

def evaluateModel(NbGame: int, modelName: str):
    for i in range(NbGame):
        score, maxTile = startTerminal(modelName)
        writeResultat(modelName, maxTile, score)
        
def main(argv):
    evaluateModel(int(argv[2]),argv[1])

if __name__ == "__main__":
    main(sys.argv)

"""
grid = Grid([[]])
matrix = [
    [256,64,0,0],
    [128,0,0,0],
    [0,0,256,0],
    [0,0,0,0],
]
grid.setMatrix(matrix)
matrix2 = grid.getMatrix()
"""