# Code for generating training data.
#
# Author: Julia Kaeppel
import pickle
from quoridor import Board
import minimax
import random

# Training data information for a board state
class StateInfo:
    # Initializes a StateInfo object.
    #
    # score_prio: Lowest score priority generated from this state. At the
    # default of zero, it means no boards have been generated from this one.
    def __init__(self, score: int=None, score_prio: int=0):
        self.score = score
        self.score_prio = score_prio

class DataSet:
    # Initializes a Dataset.
    def __init__(self):
        self.valid = dict()
        self.processed = dict()
        self.unprocessed = dict()
        self.processed_prio = 0
        key = (Board(), True)
        self.unprocessed[key] = StateInfo()

    # Generates new states up to the target size
    def gen_states(self, target_size: int):
        while len(self.valid) < target_size:
            print(len(self.valid))

            # If there are no unprocessed states, begin processing the previously
            # processed states
            if len(self.unprocessed) == 0:
                self.unprocessed = self.processed
                self.processed = dict()
            
            # Select a random state to process, removing it from the unprocessed list
            key = random.choice(list(self.unprocessed.keys()))
            state_info = self.unprocessed[key]
            del self.unprocessed[key]

            # Update score priority and calculate score
            board = key[0]
            p1_turn = key[1]
            state_info.score_prio += 1
            score, child = minimax.negamax(board, minimax.MAX_DEPTH, \
                float("-inf"), float("+inf"), p1_turn, True, state_info.score_prio)
            
            # Only store score for a priority of 1
            if state_info.score_prio == 1:
                state_info.score = score
            
            # Add state to valid set
            self.valid[key] = state_info

            # Add state to either unprocessed or processed set based on prio
            if state_info.score_prio < self.processed_prio:
                self.unprocessed[key] = state_info
            else:
                self.processed[key] = state_info

            # Update processed_prio if a new max has been reached
            if state_info.score_prio > self.processed_prio:
                self.processed_prio = state_info.score_prio
            
            # Add child to unprocessed set if not already contained
            child_key = (child, not p1_turn)
            if not (child_key in self.valid or child_key in self.unprocessed):
                self.unprocessed[child_key] = StateInfo()
