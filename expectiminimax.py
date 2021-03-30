from typing import Tuple, List
from sys import maxsize as MAX_INT
from Grid import Grid

def expectiminimax(state: Grid, depth: int, node: str, nextnode: str):
    resGrid = Grid(matrix=state.getMatrix())
    if state.isTerminal(who=node) or depth == 0:
        return state, state.utility()
    depth -= 1
    a=0
    if node == "max":
        a = -1
        for child in state.getChildren(who=node):
            childGrid = Grid(matrix=state.getMatrix())
            childGrid.move(child)
            grid, b = expectiminimax(childGrid, depth, "rand", "min")
            if a < b:
                resGrid = grid
                a = b

    elif node == "min":
        a = MAX_INT
        for child in state.getChildren(who=node):
            childGrid = Grid(matrix=state.getMatrix())
            childGrid.move(child)
            _, b = expectiminimax(childGrid, depth, "rand", "max")
            if a > b:
                a = b
    elif node == "rand":
        a = 0
        for child in state.getChildren(who=node):
            childGrid = Grid(matrix=state.getMatrix())
            childGrid.move(child)
            _, b = expectiminimax(childGrid, nextnode, "max", _)
            a += b / (state.getChildren(who=node)).len()
    return resGrid, a

def maximize(state: Grid, depth=0) :
    if state.isTerminal(who="max") or depth == 0:
        return None, state.eval_board()
    depth -= 1
    bestMove = None
    maxUtility = float('-inf')
    for move in state.getChildren(who="max"):
        childGrid = Grid(matrix=state.getMatrix())
        childGrid.move(move)
        utility = chance(childGrid, depth)
        if maxUtility < utility:
            bestMove = move
            maxUtility = utility
    return bestMove,maxUtility

def chance (state: Grid, depth: int):
    emptyCells = state.getAvailableMovesForMin()
    nbEmpty = len(emptyCells)
    if state.isTerminal(who="min") or depth == 0:
        return state.eval_board()
    depth -= 1
#Ici c'était pour incrémenter la depth et à partir d'une certaine depth on retourne directement la utility(), on fait pas récursion, c'est ptet mieux.
    # if nbEmpty >= 6 and depth >= 3:
    #     return state.utility()
    # if nbEmpty >=0 and depth >=5:
    #     return state.utility()
    # if nbEmpty == 0 :
    #     _, utility = maximize(state,depth)
    #     return utility
    possibleTiles =[]

    chance_2 = (0.9 *(1/nbEmpty))
    chance_4 = (0.1 *(1/nbEmpty))
    for emptyCell in emptyCells :
        possibleTiles.append(((emptyCell[0]-1,emptyCell[1]-1), 2, chance_2))
        possibleTiles.append(((emptyCell[0]-1,emptyCell[1]-1), 4, chance_4))
        
    utilitySum = 0

    for t in possibleTiles:
        t_grid = Grid(matrix=state.getMatrix())
        t_grid.insertTile(t[0],t[1])
        _, utility = maximize(t_grid,depth)
        utilitySum += utility *t[2]
    return utilitySum




def getBestMoveEMM(grid: Grid, depth: int = 5):
    # child, utility = expectiminimax(Grid(matrix=grid.getMatrix()), depth, "max", None)
    move, utility =  maximize(Grid(matrix=grid.getMatrix()),depth)
    return (move,utility)
