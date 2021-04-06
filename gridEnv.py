import random
from math import log2
import numpy as np
import gym
from gym import spaces
from Grid import Grid
from typing import Any, Dict, Tuple, List
import copy
import os, time, random

class GridEnv(gym.Env,Grid):

    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,4), dtype=np.float)

        self.initializeGrid()
        self.score = 0

    def reset(self) -> List[float]:
        self.initializeGrid()
        self.score = 0
        return self.observe()

    def step(self, action: int) -> Tuple[List[float], int, bool, Dict[Any,Any]]:
        reward = - self.score
        if action == 0 and self.canMoveUp():
            self.move(action)
            self.add2Or4()
            reward += 5
        elif action == 1 and self.canMoveDown():
            self.move(action)
            self.add2Or4()
            reward += 5
        elif action == 2 and self.canMoveLeft():
            self.move(action)
            self.add2Or4()
            reward += 5
        elif action == 3 and self.canMoveRight():
            self.move(action)
            self.add2Or4()
            reward += 5
        else:
            reward -= 10
        return self.observe(), (reward + self.score), self.locked(), {}
        # return reward, observation, done, info

    def observe(self) -> List[float]:
        return [[(np.log2(x)/np.log2(65536) if x != 0 else 0) for x in line] for line in self.matrix]
        #return observation

    def render(self, mode: str = "human", close: bool = False) -> None:
        os.system('cls' if os.name=='nt' else 'clear')
        for line in (self.matrix):
            print([(tile) for tile in line])

    def locked(self) -> bool:
        clone = copy.deepcopy(self)
        if clone.canMoveUp() or clone.canMoveDown() or clone.canMoveLeft() or clone.canMoveRight(): 
            return False
        return True

