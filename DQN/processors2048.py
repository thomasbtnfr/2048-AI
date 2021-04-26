import numpy as np
from game2048 import Game2048Env
from rl.core import Processor

# Transform the observed matrix to log2N representation
class Log2NNInputProcessor(Processor):
    def process_observation(self, observation):
        """
        process_observation is called by Keras-RL to pre-process each observation of the environment state
        before passing it to the DQN/neural network agent.
        """
        observation = np.reshape(observation, (4, 4))
        observation_temp = np.where(observation <= 0, 1, observation)
        processed_observation = np.log2(observation_temp)/np.log2(65536)
        return processed_observation
    
class OneHotNNInputProcessor(Processor):
    """
    OneHotNNInputProcessor is a pre-processor for the neural network input (i.e., the observation/environment state represented
    with a single 2048 board-matrix).
    It pre-processes an observation/grid by returning all the possible grids representing the board-matrices of the game in the next
    2 steps (4+16=20 grids). In particular it encodes each of these 20 grids with a one-hot encoding method, that is it represents each
    grid with a number of matrices made of 1s and 0s equal to num_one_hot_matrices (parameter passed to the constructor).
    
    Example of one hot encoding for a single grid (in our case we will have 20 grids to encode in this way):
    S: np.array([[ 2,      2,      0,      4],
                 [ 0,      0,      8,      0],
                 [ 0,      16,     0,      0],
                 [ 65536,  0,      0,      0]])
    S_onehot: np.array([[[ 0,      0,      1,      0], # Matrix 0: 1s for grid elements s=0, 0s otherwise 
                         [ 1,      1,      0,      1],
                         [ 1,      0,      1,      1],
                         [ 0,      1,      1,      1]],
                         
                         [ 1,      1,      0,      0], # Matrix 1: 1s for grid elements s=2^1=2, 0s otherwise
                         [ 0,      0,      0,      0],
                         [ 0,      0,      0,      0],
                         [ 0,      0,      0,      0]],
                         
                         [ 0,      0,      0,      1], # Matrix 2: 1s for grid elements s=2^2=4, 0s otherwise
                         [ 0,      0,      0,      0],
                         [ 0,      0,      0,      0],
                         [ 0,      0,      0,      0]],
                         
                         [ 0,      0,      0,      0], # Matrix 3: 1s for grid elements s=2^3=8, 0s otherwise
                         [ 0,      0,      1,      0],
                         [ 0,      0,      0,      0],
                         [ 0,      0,      0,      0]],
                         
                         [ 0,      0,      0,      0], # Matrix 4: 1s for grid elements s=2^4=16, 0s otherwise
                         [ 0,      0,      0,      0],
                         [ 0,      1,      0,      0],
                         [ 0,      0,      0,      0]],
                         
                         ...
                         
                         [ 0,      0,      0,      0], # Matrix 16: 1s for grid elements s=2^16=65536, 0s otherwise
                         [ 0,      0,      0,      0], # 2^16 = 65536
                         [ 0,      0,      0,      0],
                         [ 1,      0,      0,      0]]])
    """
    
    def __init__(self, num_one_hot_matrices=16, window_length=1, model="dnn"):
        """
        Check description of OneHotNNInputProcessor class
        
        Args:
             num_one_hot_matrices: number of matrices to use to encode (via one-hot encoding) each game grid.
                 Assuming that the max achievable in the 2048 game is 65536
                 this number should normally be 16.
        """
        self.num_one_hot_matrices = num_one_hot_matrices
        self.window_length = window_length
        self.model = model
        
        self.game_env = Game2048Env() 
        
        # Variables used by one_hot_encoding() function:
        self.table = {2**i:i for i in range(1,self.num_one_hot_matrices)} # dictionary storing powers of 2: {2: 1, 4: 2, 8: 3, ..., 16384: 14, 32768: 15, 65536: 16}
        self.table[0] = 0 # Add element {0: 0} to the dictionary
    
    def one_hot_encoding(self, grid):
        """
        one_hot_encoding receives a grid representing the board-matrix of the game 2048 and returns a one-hot encoding
        representation.
        Check the description of the OneHotNNInputProcessor class to see an example of such encoding.
        
        Args:
            grid: 4x4 numpy.array representing the board-matrix of the game 2048
            
        Returns:
            numpy.array containing a number equal to num_one_hot_matrices of 4x4 numpy.arrays
        """
        grid_onehot = np.zeros(shape=(self.num_one_hot_matrices, 4, 4)) # generate 16 matrix of size 4x4
        for i in range(4):
            for j in range(4):
                grid_element = grid[i, j]
                grid_onehot[self.table[grid_element],i, j]=1
        return grid_onehot

    def get_grids_next_step(self, grid):
        """
        get_grids_next_step receives a grid representing the board-matrix of the game 2048 and returns
        a list of grids representing the 4 possible grids at the next step, one for each possible movement.
        
        Args:
            grid: 4x4 numpy.array representing the board-matrix of the game 2048 
        
        Returns:
            list of 4 numpy.arrays representing the 4 possible grids at the next step, one for each possible
            movement.
        """
        grids_list = []
        for movement in range(4):
            grid_before = grid.copy()
            self.game_env.set_board(grid_before)
            try:
                _ = self.game_env.move(movement)
            except:
                pass
            grid_after = self.game_env.get_board()
            grids_list.append(grid_after)
        return grids_list

    def process_observation(self, observation):
        """
        process_observation is the interface function called by Keras-RL to pre-process each observation of the environment state
        before passing it to the DQN/neural network agent. 
        
        Args:
            observation: numpy.array representing the board-matrix of the game 2048 
        
        Returns:
            list of numpy.arrays with all the possible grids representing the board-matrices of the game in the next 2 steps, with
            each grid encoded with a one-hot encoding method.
        """
        observation = np.reshape(observation, (4, 4))
        
        grids_list_step1 = self.get_grids_next_step(observation)
        grids_list_step2 =[]
        for grid in grids_list_step1:
            grids_list_step2.append(grid) # In the NN input I give both, the 1-step and 2-step grids
            grids_temp = self.get_grids_next_step(grid)
            for grid_temp in grids_temp:
                grids_list_step2.append(grid_temp)
        grids_list = np.array([self.one_hot_encoding(grid) for grid in grids_list_step2])
        
        return grids_list
    
    def process_state_batch(self, batch):
        """
        process_state_batch processes an entire batch of states and returns it.
        It is required to reshape the NN input in case we want to use a CNN model with the one-hot encoding.
        The implementation contemplates only the case where we look at the grids of the 2 next steps (for a total
        of 4+4*4=20 grids).
        Check the comments in dqn2048.py regarding the input shape of the CNNs.
        
        Args:
            batch (list): List of states
        
        Returns:
            Processed list of states
        """
        if self.model == "cnn": # batch pre-processing only required for the cnn models
            try:
                batch = np.reshape(batch, (self.window_length, self.window_length*(4+4*4)*self.num_one_hot_matrices, 4, 4))
            except:
                batch = np.reshape(batch, (np.shape(batch)[0], self.window_length*(4+4*4)*self.num_one_hot_matrices, 4, 4))
                pass
        return batch
