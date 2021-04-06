import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from gridEnv import GridEnv

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def build_model():
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(1, 4, 4)))
    model.add(Convolution2D(4, (2, 2), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (1, 1), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (1, 1), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    return model

env = GridEnv()
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
print(nb_actions)

model = build_model()
memory = SequentialMemory(limit=1000000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])


weights_filename = 'data/weights.h5f'
checkpoint_weights_filename = 'data/weights_ch_{step}.h5f'
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]

dqn.fit(env, callbacks=callbacks, nb_steps=500000, verbose=1)
dqn.save_weights(weights_filename, overwrite=True)

testEnv = GridEnv()
testEnv.reset()

dqn.load_weights('data/weights.h5f')
dqn.test(testEnv, nb_episodes=10, visualize=True, verbose=1)