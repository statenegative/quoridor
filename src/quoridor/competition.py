# Program to run a competition between two Quoridor bots.
#
# Author: Julia Kaeppel
import subprocess
import sys
import numpy as np
from serializable_board import SerializableBoard
from quoridor import Board
import random

def main():
    # Read commandline args
    if len(sys.argv) < 5 or len(sys.argv) != 5 + int(sys.argv[2]) + int(sys.argv[4]):
        print(f"usage: python {sys.argv[0]} program1 program1_nargs program2 program2_nargs [program1_args...] [program2_args...]")
        return
    
    program1_file = sys.argv[1]
    program2_file = sys.argv[3]
    program1_nargs = int(sys.argv[2])
    program2_nargs = int(sys.argv[4])
    program1_args = sys.argv[5:5+program1_nargs]
    program2_args = sys.argv[5+program1_nargs:5+program1_nargs+program2_nargs]

    # Pick player 1 and 2
    if random.randrange(0, 2) == 0:
        player1_args = [program1_file] + program1_args
        player2_args = [program2_file] + program2_args
    else:
        player1_args = [program2_file] + program2_args
        player2_args = [program1_file] + program1_args
    
    # Create board
    board = SerializableBoard(np.full(shape=(2, 8, 8), fill_value=False, dtype=bool), \
        (4, 0), (4, 8), 10, 10, True)
    
    # Game loop
    while True:
        # Run player 1 program
        board.p1_turn = True
        player1_proc = subprocess.Popen(player1_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        player1_board, _ = player1_proc.communicate(board.write_board())
        board = SerializableBoard.read_board(player1_board)

        # Display board
        display_board = Board(board.walls, board.p1, board.p2, board.p1_walls, board.p2_walls)
        print(display_board)

        # Check for win state
        if board.p1[1] == 8:
            print(f"{player1_args[0]} wins!")
            break

        # Run player 2 program
        board.p1_turn = False
        player2_proc = subprocess.Popen(player2_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        player2_board, _ = player2_proc.communicate(board.write_board())
        board = SerializableBoard.read_board(player2_board)

        # Display board
        display_board = Board(board.walls, board.p1, board.p2, board.p1_walls, board.p2_walls)
        print(display_board)

        # Check for win state
        if board.p2[1] == 0:
            print(f"{player2_args[0]} wins!")
            break

if __name__ == "__main__":
    main()
