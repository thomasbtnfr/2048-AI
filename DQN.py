import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from gridEnv import GridEnv
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import os

env = GridEnv()

# valeurs possibles
states = env.observation_space.shape  # (16,)

# actions possibles
actions = env.action_space.n # 4

# Simulate game with env

"""
episodes = 10
scores = []
maxTiles = []
for episode in range(1, episodes+1):
    env.reset()
    done = False
    score = 0 
    
    while not done:
        os.system('cls' if os.name=='nt' else 'clear')
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    scores.append(env.score)
    maxTiles.append(env.maxValue())
print(scores)
print(maxTiles)
"""

def build_model(states, actions):
    model = Sequential()    
    print(states)
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

model = build_model(states, actions)
print(model.summary())

dqn = build_agent(model,actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env,nb_steps=50)

"""
env = GridEnv()
#valeur possible
print(env.observation_space.sample()[0])

#action possible
print(env.action_space.sample())
print(states,actions)

matrix = [
    [256,64,4,2],
    [128,16,8,0],
    [0,0,0,0],
    [0,0,0,0],
]
print(get_grids_next_step(matrix))
"""