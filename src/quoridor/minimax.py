# Minimax.
#
# Author: Julia Kaeppel
from quoridor import Board
from typing import Optional
import heapq
from sortedcontainers import SortedList

# Max depth to search
MAX_DEPTH = 2

# Class for sorting state-score pairs
class StateScore:
    def __init__(self, state: Board, score: int):
        self.state = state
        self.score = score
    
    def __eq__(self, other):
        return self.score == other.score
    
    def __lt__(self, other):
        return self.score < other.score

# Picks the best move
def pick_move(state: Board, p1_turn: bool) -> Board:
    return negamax(state, MAX_DEPTH, float("-inf"), float("+inf"), p1_turn, False, 0)[1]

# Negamax implementation
def negamax(state: Board, depth: int, alpha: float, beta: float, p1_turn: bool, gen_data: bool, score_prio: int) -> tuple[float, Optional[Board]]:
    color = 1 if p1_turn else -1

    # Check for terminal state reached
    if state.terminal():
        return (color * float("-inf"), state)
    
    # Check for max depth reached
    if depth == 0:
        return (color * heuristic(state), None)
    
    # Handle training data generation
    if gen_data:
        state_scores = SortedList()

    # Recursively find the best child state
    value = float("-inf")
    children = state.adj_states(p1_turn)
    max_state = children[0]
    for child in children:
        # Calculate child score
        score = -negamax(child, depth - 1, -beta, -alpha, not p1_turn, False, 0)[0]

        # Handle training data generation
        if gen_data:
            state_scores.add(StateScore(child, score))

        # Check for new best score
        if score > value:
            value = score
            max_state = child
        
        # Only handle alpha-beta pruning when gen_data is disabled
        if not gen_data:
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    
    # This is just here to let me know something has gone catastrophically wrong
    if max_state == None:
        print("PANIC!")
    
    # Determine whether to return a score priority
    if not gen_data:
        return (value, max_state)
    else:
        # Ensure index out of bounds error doesn't occur
        index = -min(score_prio, len(state_scores))
        state_score = state_scores[index]
        return (state_score.score, state_score.state)

# Manual heuristic which compares the shortest path distances of the two pawns
def heuristic(state: Board) -> float:
    return state.p2_dist - state.p1_dist
