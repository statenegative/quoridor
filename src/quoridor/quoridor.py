# Quoridor game implementation.
#
# Author: Julia Kaeppel and Ben McAuliffe
from enum import Enum
import numpy as np
from copy import deepcopy
import heapq
from typing import Optional

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
    # Initializes a board
    def __init__(self, walls: np.ndarray=np.full(shape=(2, 8, 8), fill_value=False, dtype=bool), \
        p1: (int, int)=(4, 0), p2: (int, int)=(4, 8), p1_walls: int=10, p2_walls: int=10):
        self.walls = walls
        self.p1 = p1
        self.p2 = p2
        self.p1_walls = p1_walls
        self.p2_walls = p2_walls

        # Precompute shortest paths and hash code
        self.p1_dist = self.shortest_path(True)
        self.p2_dist = self.shortest_path(False)
        self.hash = hash((self.walls.data.tobytes(), self.p1, self.p2, self.p1_walls, self.p2_walls))
    
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
    def adj_states(self, p1_turn: bool) -> list["Board"]:
        states = []

        # Handle movement and jumping. This is done before wall placement since
        # on average, moving is a very bad idea and will thus establish a good
        # lower bound for alpha-beta pruning.
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
            if not self.wall_adj(ap[0], ap[1], _Dir.LEFT):
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
        
        # Ensure a wall can actually be placed
        active_walls = self.p1_walls if p1_turn else self.p2_walls
        if active_walls == 0:
            return states

        # Order walls based on proximity to pawns
        walls = []
        for y in range(8):
            for x in range(8):
                prox = min(self.__prox(self.p1, x, y), self.__prox(self.p2, x, y))
                heapq.heappush(walls, (prox, x, y))

        # Place walls
        while walls:
            _, x, y = heapq.heappop(walls)

            # Horizontal wall
            if not (self.wall_adj(x, y, _Dir.UP) or
                self.wall_adj(x + 1, y, _Dir.UP) or self.walls[1][y][x]):
                # Ensure wall placement is valid
                state = self.__place_wall(p1_turn, x, y, 0)
                state.p1_dist = state.shortest_path(True)
                state.p2_dist = state.shortest_path(False)
                if state.p1_dist != None and state.p2_dist != None:
                    states.append(state)
            
            # Vertical wall
            if not (self.wall_adj(x, y, _Dir.RIGHT) or
                self.wall_adj(x, y + 1, _Dir.RIGHT) or
                self.walls[0][y][x]):
                # Ensure wall placement is valid
                state = self.__place_wall(p1_turn, x, y, 1)
                state.p1_dist = state.shortest_path(True)
                state.p2_dist = state.shortest_path(False)
                if state.p1_dist != None and state.p2_dist != None:
                    states.append(state)

        return states
    
    # Determines whether a path to the end exists
    def shortest_path(self, p1_turn: bool) -> Optional[float]:
        open_list = []
        closed_list = set()

        # Add initial position to queue
        ap = self.p1 if p1_turn else self.p2
        target = 8 if p1_turn else 0
        heapq.heappush(open_list, (abs(ap[1] - target), 0, ap[0], ap[1]))

        # Loop over queue
        while open_list:
            elem = heapq.heappop(open_list)
            dist = elem[1] + 1
            x, y = elem[2], elem[3]
            closed_list.add((x, y))
            
            # Check for completion
            if p1_turn:
                if y == 7 and not self.wall_adj(x, y, _Dir.UP):
                    return dist 
            else:
                if y == 1 and not self.wall_adj(x, y, _Dir.DOWN):
                    return dist
            
            # Move up
            if y < 8 and not self.wall_adj(x, y, _Dir.UP) and not (x, y + 1) in closed_list:
                heapq.heappush(open_list, (abs(y + 1 - target) + dist, dist, x, y + 1))
            
            # Move down
            if y > 0 and not self.wall_adj(x, y, _Dir.DOWN):
                heapq.heappush(open_list, (abs(y - 1 - target) + dist, dist, x, y - 1))
            
            # Move right
            if x < 8 and not self.wall_adj(x, y, _Dir.RIGHT) and not (x + 1, y) in closed_list:
                heapq.heappush(open_list, (abs(y - target) + dist, dist, x + 1, y))
            
            # Move left
            if x > 0 and not self.wall_adj(x, y, _Dir.LEFT) and not (x - 1, y) in closed_list:
                heapq.heappush(open_list, (abs(y - target) + dist, dist, x - 1, y))

        return None
    
    # Checks whether this state is terminal
    def terminal(self):
        return self.p1[1] == 8 or self.p2[1] == 0
    
    # Compares boards. All fields except dist fields are compared, since those
    # are guaranteed to be the same if all other fields match.
    def __eq__(self, other):
        # Check for None
        if other == None:
            return False

        return (self.walls == other.walls).all() and self.p1 == other.p1 and \
            self.p2 == other.p2 and self.p1_walls == other.p1_walls and \
            self.p2_walls == other.p2_walls
    
    def __hash__(self) -> int:
        return self.hash

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
    
    # Moves a pawn to a specified position
    def __move_pawn(self, p1_turn: bool, x: int, y: int) -> "Board":
        state = deepcopy(self)
        if p1_turn:
            state.p1 = (x, y)
            state.p1_dist = state.shortest_path(True)
        else:
            state.p2 = (x, y)
            state.p2_dist = state.shortest_path(False)
        return state
    
    # Calculates the proximity of a wall to a pawn
    def __prox(self, pawn: tuple[int, int], x: int, y: int) -> int:
        prox = 0
        
        # Handle horizontal component
        if x < pawn[0]:
            prox += pawn[0] - x - 1
        else:
            prox += x - pawn[0]

        # Handle vertical component
        if y < pawn[1]:
            prox += pawn[1] - y - 1
        else:
            prox += y - pawn[1]
        
        return prox

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
