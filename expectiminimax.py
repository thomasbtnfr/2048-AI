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
            grid, b = expectiminimax(childGrid, depth, "min", "min")
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
            _, b = expectiminimax(childGrid, depth, nextnode, _)
            a += b / (state.getChildren(who=node)).len()
    return resGrid, a


def getBestMoveEMM(grid: Grid, depth: int = 5):
    child, utility = expectiminimax(Grid(matrix=grid.getMatrix()), depth, "max", None)
    return (grid.getMoveTo(child),utility)
