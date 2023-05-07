# Quoridor game implementation.
#
# Author: Julia Kaeppel and Ben McAuliffe
from enum import Enum
import numpy as np
from copy import deepcopy

class _Dir(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3

# The walls array has shape (2, 8, 8). The first index of the first dimension
# is for horizontal walls, and the second is for vertical walls. The second
# dimension is the Y dimension, and the third dimension is the X dimension.
# (0, 0) is the bottom left corner
class Board:
    # Initializes an empty board with initial pawn placements.
    def __init__(self):
        self.walls = np.full(shape=(2, 8, 8), fill_value=False, dtype=bool)
        self.p1 = (4, 0)
        self.p2 = (4, 8)
        self.p1_walls = 10
        self.p2_walls = 10
    
    # Returns whether a wall is adjacent to a tile in a given direction.
    def wall_adj(self, x: np.uint8, y: np.uint8, dir: _Dir) -> bool:
        if dir == _Dir.UP:
            if y == 8:
                return False
            if x == 0:
                return self.walls[0][y][x]
            if x == 8:
                return self.walls[0][y][x-1]
            return self.walls[0][y][x-1] or self.walls[0][y][x]

        elif dir == _Dir.DOWN:
            if y == 0:
                return False
            if x == 0:
                return self.walls[0][y-1][x]
            if x == 8:
                return self.walls[0][y-1][x-1]
            return self.walls[0][y-1][x-1] or self.walls[0][y-1][x]
        
        elif dir == _Dir.RIGHT:
            if x == 8:
                return False
            if y == 0:
                return self.walls[1][y][x]
            if y == 8:
                return self.walls[1][y-1][x]
            return self.walls[1][y-1][x] or self.walls[1][y][x]
        
        elif dir == _Dir.LEFT:
            if x == 0:
                return False
            if y == 0:
                return self.walls[1][y][x-1]
            if y == 8:
                return self.walls[1][y-1][x-1]
            return self.walls[1][y-1][x-1] or self.walls[1][y][x-1]
    
    # Returns a list of all valid adjacent board states.
    def adj_states(self, p1_turn: bool) -> list("Board"):
        states = []

        # Place walls
        for y in range(8):
            for x in range(8):
                # Check wall count
                active_walls = self.p1_walls if p1_turn else self.p2_walls
                if active_walls == 0:
                    continue

                # Horizontal wall
                if not (self.wall_adj(x, y, _Dir.UP) or
                    self.wall_adj(x + 1, y, _Dir.UP) or self.walls[1][y][x]):
                    states.append(self.__place_wall(p1_turn, x, y, 0))
                
                # Vertical wall
                if not (self.wall_adj(x, y, _Dir.RIGHT) or
                    self.wall_adj(x, y + 1, _Dir.RIGHT) or
                    self.walls[0][y][x]):
                    states.append(self.__place_wall(p1_turn, x, y, 1))
        
        # Handle jumping
        ap, ip = (self.p1, self.p2) if p1_turn else (self.p2, self.p1)

        # Jumping up
        if ap[1] < 8:
            # Check for wall
            if not self.wall_adj(ap[0], ap[1], _Dir.UP):
                # Check for inactive pawn
                if (ap[0], ap[1] + 1) != ip:
                    states.append(self.__move_pawn(p1_turn, ap[0], ap[1] + 1))
                else:
                    # Check whether straight jump can be performed
                    if ap[1] < 7 and not self.wall_adj(ip[0], ip[1], _Dir.UP):
                        states.append(self.__move_pawn(p1_turn, ap[0], ap[1] + 2))
                    else:
                        # Check whether left diagonal jump can be performed
                        if ap[0] > 0 and not self.wall_adj(ip[0], ip[1], _Dir.LEFT):
                            states.append(self.__move_pawn(p1_turn, ap[0] - 1, ap[1] + 1))
                        # Check whether right diagonal jump can be performed
                        if ap[0] < 8 and not self.wall_adj(ip[0], ip[1], _Dir.RIGHT):
                            states.append(self.__move_pawn(p1_turn, ap[0] + 1, ap[1] + 1))
        
        # Jumping down
        if ap[1] > 0:
            # Check for wall
            if not self.wall_adj(ap[0], ap[1], _Dir.DOWN):
                # Check for inactive pawn
                if (ap[0], ap[1] - 1) != ip:
                    states.append(self.__move_pawn(p1_turn, ap[0], ap[1] - 1))
                else:
                    # Check whether straight jump can be performed
                    if ap[1] > 1 and not self.wall_adj(ip[0], ip[1], _Dir.DOWN):
                        states.append(self.__move_pawn(p1_turn, ap[0], ap[1] - 2))
                    else:
                        # Check whether left diagonal jump can be performed
                        if ap[0] > 0 and not self.wall_adj(ip[0], ip[1], _Dir.LEFT):
                            states.append(self.__move_pawn(p1_turn, ap[0] - 1, ap[1] - 1))
                        # Check whether right diagonal jump can be performed
                        if ap[0] < 8 and not self.wall_adj(ip[0], ip[1], _Dir.RIGHT):
                            states.append(self.__move_pawn(p1_turn, ap[0] + 1, ap[1] - 1))
        
        # Jumping right
        if ap[0] < 8:
            # Check for wall
            if not self.wall_adj(ap[0], ap[1], _Dir.RIGHT):
                # Check for inactive pawn
                if (ap[0] + 1, ap[1]) != ip:
                    states.append(self.__move_pawn(p1_turn, ap[0] + 1, ap[1]))
                else:
                    # Check whether straight jump can be performed
                    if ap[0] < 7 and not self.wall_adj(ip[0], ip[1], _Dir.RIGHT):
                        states.append(self.__move_pawn(p1_turn, ap[0] + 2, ap[1]))
                    else:
                        # Check whether downward diagonal jump can be performed
                        if ap[1] > 0 and not self.wall_adj(ip[0], ip[1], _Dir.DOWN):
                            states.append(self.__move_pawn(p1_turn, ap[0] + 1, ap[1] - 1))
                        # Check whether upward diagonal jump can be performed
                        if ap[1] < 8 and not self.wall_adj(ip[0], ip[1], _Dir.UP):
                            states.append(self.__move_pawn(p1_turn, ap[0] + 1, ap[1] + 1))

        # Jumping left
        if ap[0] > 0:
            # Check for wall
            if not self.wall_adj(ap[0], ap[1], _Dir.RIGHT):
                # Check for inactive pawn
                if (ap[0] - 1, ap[1]) != ip:
                    states.append(self.__move_pawn(p1_turn, ap[0] - 1, ap[1]))
                else:
                    # Check whether straight jump can be performed
                    if ap[0] > 1 and not self.wall_adj(ip[0], ip[1], _Dir.LEFT):
                        states.append(self.__move_pawn(p1_turn, ap[0] - 2, ap[1]))
                    else:
                        # Check whether downward diagonal jump can be performed
                        if ap[1] > 0 and not self.wall_adj(ip[0], ip[1], _Dir.DOWN):
                            states.append(self.__move_pawn(p1_turn, ap[0] - 1, ap[1] - 1))
                        # Check whether upward diagonal jump can be performed
                        if ap[1] < 8 and not self.wall_adj(ip[0], ip[1], _Dir.UP):
                            states.append(self.__move_pawn(p1_turn, ap[0] - 1, ap[1] + 1))

        return states
    
    # Places a wall, creating a new state
    def __place_wall(self, p1_turn: bool, x: int, y: int, alignment: int) \
        -> "Board":
        state = deepcopy(self)
        state.walls[alignment][y][x] = True
        if p1_turn:
            state.p1_walls -= 1
        else:
            state.p2_walls -= 1
        return state
    
    def __move_pawn(self, p1_turn: bool, x: int, y: int) -> "Board":
        state = deepcopy(self)
        if p1_turn:
            state.p1 = (x, y)
        else:
            state.p2 = (x, y)
        return state

    def __str__(self):
        s = ""
        for y in range(8, -1, -1):
            # Draw row of tiles
            for x in range(9):
                # Place vertical wall segment
                if x > 0:
                    if self.wall_adj(x, y, _Dir.LEFT):
                        s += '#'
                    else:
                        s += '|'
                
                # Place pawns
                if (x, y) == self.p1:
                    s += '1'
                elif (x, y) == self.p2:
                    s += '2'
                else:
                    s += '.'
            s += '\n'
            
            # Draw row of horizontal walls
            for x in range(9):
                # Place horizontal wall segment
                if y > 0:
                    if self.wall_adj(x, y, _Dir.DOWN):
                        s += '#'
                    else:
                        s += '-'
                
                # Place corner
                if x < 8 and y > 0:
                    if y > 0 and (self.walls[0][y-1][x] or
                        self.walls[1][y-1][x]):
                        s += '#'
                    else:
                        s += '+'
            if y > 0:
                s += '\n'
        
        # Add wall counts
        s = f"{s}P1: {self.p1_walls:2}     P2: {self.p2_walls:2}"
        return s
