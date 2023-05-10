# Serializable board class.
#
# Author: Julia Kaeppel
import numpy as np
import pickle
import sys

class SerializableBoard:
    # Creates a SerializableBoard.
    def __init__(self, walls: np.ndarray, p1: (int, int), p2: (int, int), p1_walls: int, p2_walls: int, p1_turn: bool):
        if walls.shape != (2, 8, 8):
            raise ValueError(f"walls of shape {walls.shape()} don't match expected shape [2, 8, 8]")

        self.walls = walls
        self.p1 = p1
        self.p2 = p2
        self.p1_walls = p1_walls
        self.p2_walls = p2_walls
        self.p1_turn = p1_turn
    
    # Writes a SerializableBoard to a binary stream.
    def write_board(self) -> bytes:
        return pickle.dumps(self)
    
    # Outputs a SerializableBoard to stdout.
    def output_board(self):
        sys.stdout.buffer.write(self.write_board())
    
    # Reads a SerializableBoard from a binary stream.
    def read_board(input: bytes) -> "SerializableBoard":
        return pickle.loads(input)
    
    # Reads a SerializableBoard from stdin.
    def input_board() -> "SerializableBoard":
        return pickle.loads(sys.stdin.buffer.read())
