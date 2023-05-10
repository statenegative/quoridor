#!/usr/bin/python3

# Quoridor bot.
#
# Author: Julia Kaeppel
from serializable_board import SerializableBoard
from quoridor import Board
import minimax

def main():
    # Read input
    sb = SerializableBoard.input_board()
    # Compute move
    board = Board(sb.walls, sb.p1, sb.p2, sb.p1_walls, sb.p2_walls)
    board = minimax.pick_move(board, sb.p1_turn)
    sb = SerializableBoard(board.walls, board.p1, board.p2, board.p1_walls, \
        board.p2_walls, not sb.p1_turn)
    # Write output
    sb.output_board()

if __name__ == "__main__":
    main()
