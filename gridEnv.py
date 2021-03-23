from Grid import Grid
import gym
from gym import spaces
from gym.utils import seeding
import random
import numpy as np
import os

class GridEnv(Grid,gym.Env):
    def __init__(self):
        self.size = 4
        # Actions we can take : up, down, left, right
        self.action_space = spaces.Discrete(4)
        # Observation space, possible values for a tile and number of tile
        self.observation_space = spaces.Box(0, 2**16, (self.size*self.size, ), dtype = np.int)
        self.matrix = []
        self.initializeGrid()
        self.state = self.matrix
        self.seed()
        self.score = 0

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Play one step of the game. Moving and adding a new tile
        score = 0

        if self.isGameOver():
            done = True
            observation = self.matrix
            reward = -5
            info = {}
            return observation, reward, done, info
        else:
            done = False

        if action == 0 and self.canMoveUp():
            score += self.up()
            reward = float(score)
            self.add2Or4()
        elif action == 1 and self.canMoveDown():
            score += self.down()
            reward = float(score)
            self.add2Or4()
        elif action == 2 and self.canMoveLeft():
            score += self.left()
            reward = float(score)
            self.add2Or4()
        elif action == 3 and self.canMoveRight():
            score += self.right()
            reward = float(score)
            self.add2Or4()
        else:
            reward = -5

        info = {}
        observation = self.matrix
        return observation, reward, done, info

    def render(self):
        # Rendering for standard output of score
        print('Score : ', self.score)
        self.printGrid()

    def reset(self):
        # Reset the game board and add two tiles
        #os.system('cls' if os.name=='nt' else 'clear')
        self.initializeGrid()
        self.score = 0
        return self.getMatrix()

    def up(self):
        scoreToReturn = 0
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
                    self.score += 2*k
                    scoreToReturn += 2*k
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
        return scoreToReturn
    
    def down(self):
        scoreToReturn = 0
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
                    self.score += 2*k
                    scoreToReturn += 2*k
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
        return scoreToReturn
    
    def left(self):
        scoreToReturn = 0
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
                    self.score += 2*k
                    scoreToReturn += 2*k
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
        return scoreToReturn
    
    def right(self):
        scoreToReturn = 0
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
                    self.score += 2*k
                    scoreToReturn += 2*k
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
        return scoreToReturn


"""
env = GridEnv()
print(env.render())
matrix = [
    [1,2,3,4],
    [5,6,7,8],
    [1,2,3,4],
    [5,6,7,8],
]
env.setMatrix(matrix)
print(not env.isGameOver())
"""