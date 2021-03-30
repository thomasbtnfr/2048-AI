from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import random
import math
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import Callback
from gridBis import Game2048Env

NeedToTrain = True

"""
#Define a 2048 environment using OpenAI gym 

# Function to create an empty board
def create_empty_board():
    # Create a new board and fill it with NaN
    board = np.empty((4,4))
    board[:] = np.NaN
    return board

# Function to add a new game piece
def add_piece(board):
    #Find indicies of all open spaces
    empties = np.empty((1,2))
    spaces = 0
    for i in range(4):
        for j in range(4):
            if math.isnan(board[i,j]):
                if empties.shape[0] == 1:
                    empties = np.array([i,j])
                    spaces = spaces + 1
                else:
                    spaces = spaces + 1
                    empties = np.vstack((empties, [i,j]))

    # Pick a random space from the list of open spaces and add a number in that location
    if spaces == 1:
        index = empties
    else:
        index = empties[random.choice(list(range(0, spaces)))]
    piece = random.choice([2,2,4])
    board[index[0], index[1]] = piece

    return board

# Function to check if game is still going
def game_check (board):
    alive = False
    for i in range(4):
        for j in range(4):
            if math.isnan(board[i,j]):
                alive = True
    return alive

# Function to Rotate the board (Makes processing moves easier)
def rotateMatrix(mat):
    N = 4
    for x in range(0, int(N / 2)):
        for y in range(x, N - x - 1):
            # store current cell in temp variable
            temp = mat[x][y]

            # move values from right to top
            mat[x][y] = mat[y][N - 1 - x]

            # move values from bottom to right
            mat[y][N - 1 - x] = mat[N - 1 - x][N - 1 - y]

            # move values from left to bottom
            mat[N - 1 - x][N - 1 - y] = mat[N - 1 - y][x]

            # assign temp to left
            mat[N - 1 - y][x] = temp

    return mat

# Functions to process moves
def move_down(board):

    # Column by column, check for empties, combine, and move
    for i in range(4):

        # Find values in a column
        pieces = board[:,i]
        pieces = pieces[np.logical_not(np.isnan(pieces))]


        # Check for any matchings and combine
        if len(pieces) > 1:
            for j in range(len(pieces)-1, 0, -1):
                if pieces[j] == pieces[j-1]:
                    pieces[j] = (pieces[j] * 2)
                    pieces[j - 1] = float('NaN')

        # Place combined numbers
        pieces = pieces[np.logical_not(np.isnan(pieces))]
        if len(pieces) == 4:
            board[:, i] = pieces.reshape(-1)
        else:
            spaces = np.empty(((4-len(pieces)), 1))
            spaces[:] = np.NaN
            pieces = np.vstack((spaces, pieces.reshape(-1,1)))
            board[:, i] =pieces.reshape(-1)
    return board
def move_left(board):
    board = rotateMatrix(board)
    board = move_down(board)
    board = rotateMatrix(board)
    board = rotateMatrix(board)
    board = rotateMatrix(board)
    return board
def move_up(board):
    board = rotateMatrix(board)
    board = rotateMatrix(board)
    board = move_down(board)
    board = rotateMatrix(board)
    board = rotateMatrix(board)
    return board
def move_right(board):
    board = rotateMatrix(board)
    board = rotateMatrix(board)
    board = rotateMatrix(board)
    board = move_down(board)
    board = rotateMatrix(board)
    return board

# Function to reformat board
def to_observation(board):
    board = np.nan_to_num(board, copy=True, nan=0.0, posinf=None, neginf=None)
    board = board.reshape(1,-1)
    board = np.squeeze(board, axis=0)
    return board

#Function to format board as a matrix
def to_matrix (board):
    board = board.reshape(4,4)
    board[board == 0] = np.nan
    return board

# Function to check whether two max pieces are adjacent
def max_pieces_adjacent(board):

    #Locate two highest numbers
    board = np.nan_to_num(board, copy=True, nan=0.0, posinf=None, neginf=None)
    max_ind = np.unravel_index(board.argmax(), board.shape)
    board[max_ind] = 0
    second_max_ind = np.unravel_index(board.argmax(), board.shape)

    #Determine if positions are adjacent
    distance = abs(max_ind[0] - second_max_ind[0]) + abs(max_ind[1] - second_max_ind[1])
    if distance <= 1:
        adjacent = True
    else:
        adjacent = False

    return adjacent


# Define our custom environment for the agent to operate in
class GameEnv(Env):

    def __init__(self):
        # Agent can move up, down, left and right
        self.action_space = Discrete(4)
        # Observation space is the value of each boardspace
        self.observation_space = MultiDiscrete( [99999,99999,99999,99999,
                                                 99999,99999,99999,99999,
                                                 99999,99999,99999,99999,
                                                 99999,99999,99999,99999] )
        # Set start point by drawing two cards
        board = create_empty_board()
        board = add_piece(board)
        board = add_piece(board)
        state = to_observation(board)
        self.state = np.reshape(state, (1, state.shape[0]))
        print(self.action_space.shape)
        print(self.observation_space.shape)
        print(self.state.shape)

    def step(self, action):

        # Convert the board to a matrix from the environment's state
        board = to_matrix(self.state)
        pieces_before = len(self.state[self.state > 0])

        # Apply action
        # 0: Move Down
        # 1: Move Up
        # 2: Move Right
        # 3: Move Left
        if action == 0:
            board = move_down(board)
        elif action == 1:
            board = move_up(board)
        elif action == 2:
            board = move_right(board)
        else:
            board = move_left(board)


        # Reward agent for (1) Combining pieces, (2) Keeping the highest piece in a corner, and (3) Keeping the highest
        # two pieces adjacent

        reward = 0

        post_move_state = to_observation(board)
        pieces_after = len(post_move_state[post_move_state>0])
        if pieces_after < pieces_before:
            reward += 1

        max_index = np.amax(to_observation(board))
        if max_index == 0 or max_index == 4 or max_index ==12 or max_index == 15:
            reward += 1

        if max_pieces_adjacent(board):
            reward += 1


        # Check if game state is still alive

        done = True
        if game_check(board):
            board = add_piece(board)
            done = False

        # If there aren't any empty spaces, check if there is a move to salvage the run and make it
        else:
            if game_check(move_down(board)):
                board = move_down(board)
                board = add_piece(board)
                done = False
            if game_check(move_up(board)):
                board = move_up(board)
                board = add_piece(board)
                done = False
            if game_check(move_left(board)):
                board = move_left(board)
                board = add_piece(board)
                done = False
            if game_check(move_right(board)):
                board = move_right(board)
                board = add_piece(board)
                done = False

        # Set the game state in the correct format
        state = to_observation(board)
        self.state = state

        # Set placeholder for info
        info = {}

        return self.state, reward, done, info


    def render(self):
        print(to_matrix(self.state))

    def reset(self):
        # Reset agent's hand
        board = create_empty_board()
        board = add_piece(add_piece(board))
        state = to_observation(board)
        self.state = state
        return self.state
"""
# Create training environment
trainEnv = Game2048Env()
testEnv = Game2048Env()
testEnv.reset()


# Test environment with random inputs from action_space
episodes = 20
for episode in range(1, episodes+1):
    state = trainEnv.reset()
    done = False
    score = 0
    steps = 0

    while not done:
        action = trainEnv.action_space.sample()
        n_state, reward, done, info, = trainEnv.step(action)
        steps += 1
        score = score + reward
    trainEnv.render()
    print('Episode: {0} Score: {1} Steps: {2}'.format(episode, score, steps))


#Create a Deep Learning Model with Keras

#Define environment's action and observation space to guide model development
states = trainEnv.observation_space.shape
actions = trainEnv.action_space.n

def build_model(states, actions):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions, activation='sigmoid'))
    return model

model = build_model(states, actions)

print(model.summary())


#Build Agent with Keras RL and TensorFlow


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 200000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions,
                   nb_steps_warmup=200, target_model_update=5e-3) #had to update dqn.py line59 -- added extra dimension
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

if NeedToTrain:
    dqn.fit(trainEnv, nb_steps=500000, visualize=False, verbose=1)
    dqn.save_weights('2048dqn_weights.h5f', overwrite=True)


#Test agent 

dqn.load_weights('2048dqn_weights.h5f')
_ = dqn.test(testEnv, nb_episodes=20, visualize = False, verbose=1)
