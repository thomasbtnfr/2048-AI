import os
import numpy as np
import random
from game2048 import Game2048Env

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from keras.layers.merge import concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from callbacks2048 import TrainEpisodeLogger2048, TestLogger2048
from processors2048 import Log2NNInputProcessor, OneHotNNInputProcessor

# CHOOSE TRAIN / TEST MODE
# TRAIN_TEST_MODE = 'train'
TRAIN_TEST_MODE = 'test'

MODEL_TYPE = 'dnn'

# We provide the future N_FUTURE_STATES environment states to the DNN network
N_FUTURE_STEPS = 2
def n_future_states(N_FUTURE_STEPS):
    res = 0
    for i in range(1,N_FUTURE_STEPS+1):
        res += 4**i
    return res
N_FUTURE_STATES = n_future_states(N_FUTURE_STEPS)
print('FUTURE STATES',N_FUTURE_STATES)

NUM_ACTIONS_OUTPUT_NN = 4 # number of possible actions ( = number of neurons for the output layer)
WINDOW_LENGTH = 1
INPUT_SHAPE = (4, 4) # game-grid/matrix size (input shape for the neural network)

# Decide the pre-processing method for the neural network inputs:
#PREPROC="log2" 
PREPROC="onehot2steps"
NUM_ONE_HOT_MAT = 16 # number of matrices to use for encoding each game-grid

# Set the training hyperparameters:
NB_STEPS_ANNEALED = int(1e5) # number of steps used in LinearAnnealedPolicy()
NB_STEPS_WARMUP = 5000 # number of steps to fill memory before training
TARGET_MODEL_UPDATE = 1000

def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=INPUT_SHAPE_DNN))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=4, activation='linear'))
    print(model.summary())
    return model

env = Game2048Env()
np.random.seed(123)
env.seed(123)

processor = OneHotNNInputProcessor(num_one_hot_matrices=NUM_ONE_HOT_MAT, n_future_steps = N_FUTURE_STEPS)
INPUT_SHAPE_DNN = (WINDOW_LENGTH, N_FUTURE_STATES, NUM_ONE_HOT_MAT,) + INPUT_SHAPE

model = build_model()

memory = SequentialMemory(limit=1000000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1000000)
TRAIN_POLICY = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.05, value_min=0.05, value_test=0.01, nb_steps=NB_STEPS_ANNEALED)
TEST_POLICY = EpsGreedyQPolicy(eps=.01) # to reduce the number of illegal moves

dqn = DQNAgent(model=model, nb_actions=NUM_ACTIONS_OUTPUT_NN, test_policy=TEST_POLICY, policy=TRAIN_POLICY, memory=memory, processor=processor,
                    nb_steps_warmup=NB_STEPS_WARMUP, gamma=.99, target_model_update=TARGET_MODEL_UPDATE, train_interval=4, delta_clip=1.)
    
dqn.compile(Adam(lr=.00025), metrics=['mse'])


weights_filename = 'data/weights.h5f'
checkpoint_weights_filename = 'data/weights_ch_{step}.h5f'
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
callbacks += [TrainEpisodeLogger2048()]

if TRAIN_TEST_MODE == 'train':
    dqn.fit(env, callbacks=callbacks, nb_steps=100000, verbose=1)
    dqn.save_weights(weights_filename, overwrite=True)
else:
    dqn.load_weights('data/DQN_data_weights_ch_500000.h5f')
    _callbacks = [TestLogger2048()] 
    dqn.test(env, nb_episodes=1, visualize=True, callbacks=_callbacks)
