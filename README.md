# AI Project S8 INSA - 2048

Implementation of the 2048 game logic and different AI algorithms in Python.

Our goal was to implement several algorithms and to compare them.

## Algorithms :

1. ### Minmax

   The minmax algorithm is used for games between opponents. To implement it, we consider two players: max and min. Max must maximize the score and corresponds to the player. He performs at each turn one of the 4 possible actions. Min corresponds to the computer that places a 2 or 4 on the grid each turn. It is not really an opponent, because it plays randomly but for this algorithm we considered it as an opponent and the results are interisting.
2. ### Deep Reinforcement Learning (DQN)

   Deep q-learning is a reinforcement learning algorithm that looks for the best action to take based on a current state. The expected reward of an action at a step is called Q-value. This is stored in a table for each state, future state tuple. We use neural networks in addition because it is complicated to track all the possible combinations of states. So instead of using only Q-values we approximate them with neural networks, which gives the Deep Q-Network, known as DQN.

   Our implementation is based on this one : https://github.com/SergioIommi/DQN-2048. We use our own environment based on all our algorithms. We also added the feature to provide the neural network with the n future states as input.
3. ### Expectimax

   TODO
4. ### Supervised Learning

   To develop supervised learning, we stored states and the associated action in memory during games.  This algorithm was developed before the DQN to discover more easily how to implement a neural network for the 2048 game. We quickly noticed that the results were not very good because we did not have enough data. That's why we decided to switch to reinforcement learning.
5. ### MCTS : Monte-Carlo Tree-Search

## Run an AI algorithm in the terminal or in web browser

All except DQN : 

```
python game_loop.py name_algo nb_games --remote

name_algo can be : 
- minmax
- mcts
- emm
- supervised

--remote is optional, it allows to launch the 2048 game directly online with the selected algorithm

--help : presents the different possibilities
```

DQN : 

```
cd DQN
python dqn2048.py

For train/test, change the value of the boolean TRAIN_TEST_MODE
```

## Results
