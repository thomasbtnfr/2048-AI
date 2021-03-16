from copy import deepcopy
from typing import Tuple, List
import math
import random
import sys, time, os

class Grid:
    
    def __init__(self, matrix):
        self.setMatrix(matrix)
    
    # equal operator | a == b <=> A.__eq__(b)
    def __eq__(self, other) -> bool:
        for i in range(4):
            for j in range(4):
                if self.matrix[i][j] != other.matrix[i][j]:
                    return False
        return True
    
    def setMatrix(self, matrix):
        self.matrix = deepcopy(matrix)
    
    def getMatrix(self) -> List[List]:
        return deepcopy(self.matrix)
    
    def placeTile(self, row: int, col: int, tile: int):
        self.matrix[row-1][col-1] = tile
    
    def utility(self) -> int:
        """
        count = 0
        sum = 0
        for i in range(4):
            for j in range(4):
                sum += self.matrix[i][j]
                if self.matrix[i][j] != 0:
                    count += 1
        return int(sum/count)
        """
        monoWeight = 2.0
        emptyWeight = 2.7
        maxWeight = 1.0
        # smoothWeight = 0.1

        return self.maxValueCorner() * self.maxValue() / 3 + self.monotonicity() * monoWeight + self.nbEmpty() * emptyWeight + self.maxValue() * maxWeight

    def printGrid(self):
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.matrix]))

    def initializeGrid(self):
        self.matrix = [[0 for col in range(4)] for row in range(4)]

    def add2Or4(self):
        emptyCell = []
        for i in range(4):
            for j in range(4):
                if self.matrix[i][j] == 0:
                    emptyCell.append((i,j))
        cell = random.choice(emptyCell)
        self.matrix[cell[0]][cell[1]] = 2 if random.random() < 0.9 else 4

    def canMoveUp(self) -> bool:
        for j in range(4):
            k = -1
            for i in range(3, -1, -1):
                if self.matrix[i][j] > 0:
                    k = i
                    break
            if k > -1:
                for i in range(k, 0, -1):
                    if self.matrix[i-1][j] == 0 or self.matrix[i][j] == self.matrix[i-1][j]:
                        return True
        return False

    def canMoveDown(self) -> bool:
        for j in range(4):
            k = -1
            for i in range(4):
                if self.matrix[i][j] > 0:
                    k = i
                    break
            if k > -1:
                for i in range(k, 3):
                    if self.matrix[i+1][j] == 0 or self.matrix[i][j] == self.matrix[i+1][j]:
                        return True
        return False

    def canMoveLeft(self) -> bool:
        for i in range(4):
            k = -1
            for j in range(3, -1, -1):
                if self.matrix[i][j] > 0:
                    k = j
                    break
            if k > -1:
                for j in range(k, 0, -1):
                    if self.matrix[i][j-1] == 0 or self.matrix[i][j] == self.matrix[i][j-1]:
                        return True
        return False

    def canMoveRight(self) -> bool:
        for i in range(4):
            k = -1
            for j in range(4):
                if self.matrix[i][j] > 0:
                    k = j
                    break
            if k > -1:
                for j in range(k, 3):
                    if self.matrix[i][j+1] == 0 or self.matrix[i][j] == self.matrix[i][j+1]:
                        return True
        return False
    
    def getAvailableMovesForMax(self) -> List[int]:
        moves = []

        if self.canMoveUp():
            moves.append(0)
        if self.canMoveDown():
            moves.append(1)
        if self.canMoveLeft():
            moves.append(2)
        if self.canMoveRight():
            moves.append(3)
        
        return moves
    
    def getAvailableMovesForMin(self) -> List[Tuple[int]]:
        places = []
        for i in range(4):
            for j in range(4):
                if self.matrix[i][j] == 0:
                    places.append((i+1, j+1, 2))
                    places.append((i+1, j+1, 4))
        return places
    
    def getChildren(self, who: str) -> List:
        if who == "max":
            return self.getAvailableMovesForMax()
        elif who == "min":
            return self.getAvailableMovesForMin()
    
    # 2 cases : player max and min
    # we separate the 2 cases because a max player can play sometimes even if the grid is full thanks to a merge
    def isTerminal(self, who: str) -> bool:
        if who == "max":
            if self.canMoveUp():
                return False
            if self.canMoveDown():
                return False
            if self.canMoveLeft():
                return False
            if self.canMoveRight():
                return False
            return True
        elif who == "min":
            for i in range(4):
                for j in range(4):
                    if self.matrix[i][j] == 0:
                        return False
            return True
    
    def isGameOver(self) -> bool:
        return self.isTerminal(who="max")
    
    def up(self):
        for j in range(4):
            w = 0
            k = 0
            for i in range(4):
                if self.matrix[i][j] == 0:
                    continue
                if k == 0:
                    k = self.matrix[i][j]
                elif k == self.matrix[i][j]:
                    self.matrix[w][j] = 2*k
                    w += 1
                    k = 0
                else:
                    self.matrix[w][j] = k
                    w += 1
                    k = self.matrix[i][j]
            if k != 0:
                self.matrix[w][j] = k
                w += 1
            for i in range(w, 4):
                self.matrix[i][j] = 0
    
    def down(self):
        for j in range(4):
            w = 3
            k = 0
            for i in range(3, -1, -1):
                if self.matrix[i][j] == 0:
                    continue
                if k == 0:
                    k = self.matrix[i][j]
                elif k == self.matrix[i][j]:
                    self.matrix[w][j] = 2*k
                    w -= 1
                    k = 0
                else:
                    self.matrix[w][j] = k
                    w -= 1
                    k = self.matrix[i][j]
            if k != 0:
                self.matrix[w][j] = k
                w -= 1
            for i in range(w+1):
                self.matrix[i][j] = 0
    
    def left(self):
        for i in range(4):
            w = 0
            k = 0
            for j in range(4):
                if self.matrix[i][j] == 0:
                    continue
                if k == 0:
                    k = self.matrix[i][j]
                elif k == self.matrix[i][j]:
                    self.matrix[i][w] = 2*k
                    w += 1
                    k = 0
                else:
                    self.matrix[i][w] = k
                    w += 1
                    k = self.matrix[i][j]
            if k != 0:
                self.matrix[i][w] = k
                w += 1
            for j in range(w, 4):
                self.matrix[i][j] = 0
    
    def right(self):
        for i in range(4):
            w = 3
            k = 0
            for j in range(3, -1, -1):
                if self.matrix[i][j] == 0:
                    continue
                if k == 0:
                    k = self.matrix[i][j]
                elif k == self.matrix[i][j]:
                    self.matrix[i][w] = 2*k
                    w -= 1
                    k = 0
                else:
                    self.matrix[i][w] = k
                    w -= 1
                    k = self.matrix[i][j]
            if k != 0:
                self.matrix[i][w] = k
                w -= 1
            for j in range(w+1):
                self.matrix[i][j] = 0
    
    def move(self, mv: int) -> None:
        if mv == 0:
            self.up()
        elif mv == 1:
            self.down()
        elif mv == 2:
            self.left()
        else:
            self.right()
    
    def getMoveTo(self, child: 'Grid') -> int:
        if self.canMoveUp():
            g = Grid(matrix=self.getMatrix())
            g.up()
            if g == child:
                return 0
        if self.canMoveDown():
            g = Grid(matrix=self.getMatrix())
            g.down()
            if g == child:
                return 1
        if self.canMoveLeft():
            g = Grid(matrix=self.getMatrix())
            g.left()
            if g == child:
                return 2
        return 3

    def nbEmpty(self) -> int:
        counter = 0
        for i in range(4):
            for j in range(4):
                if self.matrix[i][j] == 0:
                    counter += 1
        return counter
    
    def maxValue(self) -> int:
        maxValue = 0
        for i in range(4):
            for j in range(4):
                if self.matrix[i][j] > maxValue:
                    maxValue = self.matrix[i][j]
        return maxValue
    
    def maxValueCorner(self) -> bool:
        maxValue = (0,(0,0))
        for i in range(4):
            for j in range(4):
                if self.matrix[i][j] > maxValue[0]:
                    maxValue = (self.matrix[i][j],(i,j))
        if maxValue[1] == (0,0) or maxValue[1] == (3,0) or maxValue[1] == (0,3) or maxValue[1] == (3,3):
            return True
        else:
            return False
        
                    
    def monotonicity(self) -> int:
        totals = [0,0,0,0]
        
        # up/down 
        for i in range(4):
            current = 0
            next = current+1
            while next < 4:
                while (next<4 and self.matrix[i][next] == 0):
                    next+=1
                if next >= 4:
                    next -= 1
                if self.matrix[i][current] != 0:
                    currentValue = math.log(self.matrix[i][current]) / math.log(2)
                else:
                    currentValue = 0
                if self.matrix[i][next] != 0:
                    nextValue = math.log(self.matrix[i][next]) / math.log(2)
                else:
                    nextValue = 0
                if currentValue > nextValue:
                    totals[0] += nextValue - currentValue
                elif nextValue > currentValue:
                    totals[1] += currentValue - nextValue
                current = next
                next+=1
        
        # left/right
        for j in range(4):
            current = 0
            next = current+1
            while next < 4:
                while (next<4 and self.matrix[next][j] == 0):
                    next+=1
                if next >= 4:
                    next -= 1
                if self.matrix[current][j] != 0:
                    currentValue = math.log(self.matrix[current][j]) / math.log(2)
                else:
                    currentValue = 0
                if self.matrix[next][j] != 0:
                    nextValue = math.log(self.matrix[next][j]) / math.log(2)
                else:
                    nextValue = 0
                if currentValue > nextValue:
                    totals[2] += nextValue - currentValue
                elif nextValue > currentValue:
                    totals[3] += currentValue - nextValue
                current = next
                next+=1
                
        return max(totals[0], totals[1]) + max(totals[2], totals[3])