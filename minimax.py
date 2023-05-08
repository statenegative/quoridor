# Minimax.
#
# Author: Julia Kaeppel
from quoridor import Board
from typing import Optional
import heapq

# Max depth to search
MAX_DEPTH = 2

# Picks the best move
def pick_move(state: Board, p1_turn: bool) -> Board:
    return negamax(state, MAX_DEPTH, float("-inf"), float("+inf"), p1_turn)[1]

# Negamax implementation
def negamax(state: Board, depth: int, alpha: float, beta: float, p1_turn: bool) -> tuple[float, Optional[Board]]:
    color = 1 if p1_turn else -1

    # Check for terminal state reached
    if state.terminal():
        return (color * float("+inf"), state)
    
    # Check for max depth reached
    if depth == 0:
        return (color * heuristic(state), None)
    
    # Handle training data generation
    if depth == MAX_DEPTH:
        scores = []

    # Recursively find the best child state
    value = float("-inf")
    children = state.adj_states(p1_turn)
    max_state = children[0]
    for child in children:
        # Calculate child score
        score = -negamax(child, depth - 1, -beta, -alpha, not p1_turn)[0]

        # Check for new best score
        if score > value:
            value = score
            max_state = child
        
        # Handle alpha-beta pruning
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    
    # This is just here to let me know something has gone catastrophically wrong
    if max_state == None:
        print("PANIC!")
    
    return (value, max_state)

# Manual heuristic which compares the shortest path distances of the two pawns
def heuristic(state: Board) -> float:
    return state.p2_dist - state.p1_dist

def main():
    board = Board()
    
    while not board.terminal():
        # Player 1 move
        board = pick_move(board, True)
        print(board)
        print()
        if board.terminal():
            break
            
        # Player 2 move
        board = pick_move(board, False)
        print(board)
        print()

if __name__ == "__main__":
    main()
