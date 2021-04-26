import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import sys, os
sys.path.append("..")
from Grid import Grid

class IllegalMove(Exception):
    pass

# Environment useful for the DQN and Gym, based on Grid
class Game2048Env(gym.Env,Grid):

    def __init__(self):
        self.score = 0
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 2**16, (4 * 4, ), dtype=np.int)
        self.reward_range = (0., float(2**16))
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        score = 0
        done = None
        try:
            self.movePossible(action)
            score = float(self.move(action))
            self.score += score
            assert score <= 2**16
            self.add2Or4()
            done = self.isGameOver()
            reward = float(score)
        except IllegalMove as e:
            done = False
            reward = 0.

        observation = self.matrix
        
        # info is useful for testing and for monitoring the agent (via callback functions) while it is training
        info = {"max_tile": self.maxValue()}        
        
        return observation, reward, done, info

    def movePossible(self, action):
        if (action == 0 and self.canMoveUp()) or (action == 1 and self.canMoveDown()) or (action == 2 and self.canMoveLeft()) or (action == 3 and self.canMoveRight()):
            pass
        else:
            raise IllegalMove

    def reset(self):
        self.initializeGrid()
        self.score = 0
        return self.matrix

    def render(self, mode: str = 'human', close: bool = False):
        os.system('cls' if os.name=='nt' else 'clear')
        for line in (self.matrix):
            print([(tile) for tile in line])

    def get_board(self):
        return self.matrix

    def set_board(self, newMatrix):
        self.matrix = newMatrix
