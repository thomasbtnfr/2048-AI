from Grid import Grid
import numpy as np
import random

def mcts_move(grid: Grid, searches_per_move, search_length):
    possible_first_moves = grid.getAvailableMovesForMax()
    first_move_scores = np.zeros(4)

    for first_move in possible_first_moves:
        gridFirstMove = Grid(matrix=grid.getMatrix())
        score = gridFirstMove.move(first_move)
        gridFirstMove.add2Or4()
        first_move_scores[first_move] += score

        for _ in range(searches_per_move):
            move_number = 1
            search_board = Grid(matrix=gridFirstMove.getMatrix())
            
            while move_number < search_length:
                possible_moves = search_board.getAvailableMovesForMax()
                if (len(possible_moves) != 0):
                    random_move = random.choice(possible_moves)
                    score = search_board.move(random_move)
                    first_move_scores[first_move] += score
                    search_board.add2Or4()
                move_number += 1

    best_move_index = np.argmax(first_move_scores)
    return best_move_index


"""
grid = Grid([[]])
matrix = [
    [2,256,2,2],
    [128,8,4,32],
    [2,256,128,16],
    [4,2,2048,2048],
]
grid.setMatrix(matrix)
print(mcts_move(grid,4,1))
"""
