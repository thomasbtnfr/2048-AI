from Grid import Grid
import keyboard
import os
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import ast

def writeResultat(fileName: str, grid, move):
    resultats = []
    resultats.append(str(grid) + ';' + str(move) + '\n')
    with open(fileName + '.txt', 'a') as f:
        f.write(''.join(resultats))

def flattenMatrix(matrix):
    res = []
    for row in matrix:
        for elem in row:
            res.append(elem)
    return res

def generateDataPlaying():
    moves_count = 1
    gridList = []
    moveList = []
    Key = {'Z':0,'S':1,'Q':2,'D':3,'z':0,'s':1,'q':2,'d':3}

    grid = Grid([[]])
    grid.initializeGrid() 

    while True:
        grid.printGrid()
        if grid.isGameOver() or moves_count == 7:
            print("Unfortunately, I lost the game.")
            break
        key = input('Up : Z | Down : S | Left : Q | Right : D')
        moveCode = Key[key]
        while not moveCode in grid.getAvailableMovesForMax():
            key = input('Impossible move | Up : Z | Down : S | Left : Q | Right : D')
            moveCode = Key[key]

        gridList.append(np.array(flattenMatrix(grid.getMatrix())))
        tmp = [0,0,0,0]
        tmp[moveCode] = 1
        writeResultat('supervisedData',flattenMatrix(grid.getMatrix()), tmp)
        moveList.append(tmp)
        grid.move(moveCode)
        grid.add2Or4()
        os.system('cls' if os.name=='nt' else 'clear')
        print(f'Move #{moves_count} | Score {grid.score}')
        moves_count += 1

    return gridList, moveList

def build_model():
    model = Sequential()    
    model.add(tf.keras.layers.Dense(256, activation="relu", input_dim=16))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(4, activation="softmax"))
    return model

def startTerminalSupervised(model):
    moves_str = ['UP', 'DOWN', 'LEFT', 'RIGHT']
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


def readSupervisedData():
    colnames = ['Grid','Move']
    res = pd.read_csv('supervisedData' + '.txt',names = colnames, sep=';')
    gridList = np.array(res['Grid'].tolist())
    moveList = np.array(res['Move'].tolist())

    tmp = []
    for elem in gridList:
        tmp.append(ast.literal_eval(elem))

    gridList = np.array(tmp)

    tmp = []
    for elem in moveList:
        tmp.append(ast.literal_eval(elem))

    moveList = np.array(tmp)
    return gridList, moveList

def trainModel(model):
    gridList, moveList = readSupervisedData()

    X_train,X_test,y_train,y_test = train_test_split(gridList,moveList, test_size = 0.2, random_state=42)
    
    print(model.summary())
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy','categorical_accuracy'])

    model.fit(X_train, y_train, epochs=200, batch_size=32)
    model.save('supervisedModel')
    res = model.evaluate(X_test, y_test, verbose = 1)
    print("test loss, test acc, test categorical:",res)



# model = build_model()
# trainModel(model)

model = tf.keras.models.load_model('supervisedModel')
startTerminalSupervised(model)