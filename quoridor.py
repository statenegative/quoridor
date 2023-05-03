# Quoridor game implementation.
#
# Author: Julia Kaeppel and Ben McAuliffe
from enum import Enum
import numpy as np

class _Direction(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3

# The walls array has shape (2, 8, 8). The first index of the first dimension
# is for horizontal walls, and the second is for vertical walls. The second
# dimension is the Y dimension, and the third dimension is the X dimension.s
#(0,0) is the bottom left corner
class Board:
    # Initializes an empty board with initial pawn placements.
    def __init__(self):
        self.walls = np.full(shape=(2, 8, 8), fill_value=False, dtype=bool)
        self.p1 = (4, 0)
        self.p2 = (4, 8)
        self.p1_walls = 10
        self.p2_walls = 10
    
    # Returns whether a wall is adjacent to a tile in a given direction.
    def wall_adj(self, x: np.uint8, y: np.uint8, dir: _Direction) -> bool:
        if dir == _Direction.UP:
            if y == 8:
                return False
            if x == 0:
                return self.walls[0][y][x]
            if x == 8:
                return self.walls[0][y][x-1]
            return self.walls[0][y][x-1] or self.walls[0][y][x]

        elif dir == _Direction.DOWN:
            if y == 0:
                return False
            if x == 0:
                return self.walls[0][y-1][x]
            if x == 8:
                return self.walls[0][y-1][x-1]
            return self.walls[0][y-1][x-1] or self.walls[0][y-1][x]
        
        elif dir == _Direction.RIGHT:
            if x == 8:
                return False
            if y == 0:
                return self.walls[1][y][x]
            if y == 8:
                return self.walls[1][y-1][x]
            return self.walls[1][y-1][x] or self.walls[1][y][x]
        
        elif dir == _Direction.LEFT:
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
        if (p1_turn and self.p1_walls > 0) or (not p1_turn and self.p2_walls > 0):
            for y in range(8):
                for x in range(8):
                    # Horizontal wall
                    if (not self.walls[1][y][x] and not self.walls[0][y][x] and 
                        x > 0 and not self.walls[0][y][x-1] and 
                        x < 7 and not self.walls[0][y][x+1]):
                        # Add state
                        state = self.deepcopy()

                        # Place wall and update wall count
                        state.walls[0][y][x] = True
                        if p1_turn:
                            state.p1_walls -= 1
                        else:
                            state.p2_walls -= 1

                        states.append(state)
                    
                    # Vertical wall
                    if (not self.walls[0][y][x] and not self.walls[1][y][x] and 
                        y > 0 and not self.walls[1][y-1][x] and 
                        y < 7 and not self.walls[1][y+1][x]):
                        # Add state
                        states.append(self.deepcopy())

                        # Place wall and update wall count
                        states[-1].walls[1][y][x] = True
                        if p1_turn:
                            states[-1].p1_walls -= 1
                        else:
                            states[-1].p2_walls -= 1
        
        # Pawn movement
        # Determine active and inactive pawn
        ap, ip = self.p1, self.p2 if p1_turn else self.p2, self.p1 #tuples (x,y)

        # Moving up
        if ap[1] < 8:
            if not self.wall_adj(ap[0], ap[1], _Direction.UP):
                if (ap[0], ap[1] + 1) != ip:
                    # Add state
                    states.append(self.deepcopy())
                    if p1_turn:
                        states[-1].p1 = (ap[0], ap[1] + 1)
                    else:
                        states[-1].p2 = (ap[0], ap[1] + 1)
                else:
                    #jump handeling
                    #Upward jump
                    if (ap[0], ap[1] + 1) == ip and not self.wall_adj(ap[0], ap[1]+1, _Direction.UP) and not (ip[1] == 0 or ip[1] == 8):
                        states.append(self.deepcopy())
                        if p1_turn:
                            states[-1].p1 = (ap[0], ap[1] + 2)
                        else:
                            states[-1].p2 = (ap[0], ap[1] + 2)
                    #L jump UP and to the side
                    if (ap[0], ap[1] + 1) == ip and self.wall_adj(ap[0], ap[1]+1, _Direction.UP) and not (ip[1] == 0 or ip[1] == 8):
                        #Up and Right
                        if not ip[0] == 8 and not self.wall_adj(ap[0], ap[1]+1, _Direction.RIGHT):
                            states.append(self.deepcopy())
                            if p1_turn:
                                states[-1].p1 = (ap[0] + 1, ap[1] + 1)
                            else:
                                states[-1].p2 = (ap[0] + 1, ap[1] + 1)
                        #Up and Left
                        if not ip[0] == 0 and not self.wall_adj(ap[0], ap[1]+1, _Direction.LEFT):
                            states.append(self.deepcopy())
                            if p1_turn:
                                states[-1].p1 = (ap[0] - 1, ap[1] + 1)
                            else:
                                states[-1].p2 = (ap[0] - 1, ap[1] + 1)
        # Moving down
        if ap[1] < 8:
            if not self.wall_adj(ap[0], ap[1], _Direction.DOWN):
                if (ap[0], ap[1] - 1) != ip:
                    # Add state
                    states.append(self.deepcopy())
                    if p1_turn:
                        states[-1].p1 = (ap[0], ap[1] - 1)
                    else:
                        states[-1].p2 = (ap[0], ap[1] - 1)
                else:
                    #Straight jump handeling
                    #Downward jump
                    if (ap[0], ap[1] + 1) == ip and not self.wall_adj(ap[0], ap[1] - 1, _Direction.DOWN) and not (ip[1] == 0 or ip[1] == 8):
                        states.append(self.deepcopy())
                        if p1_turn:
                            states[-1].p1 = (ap[0], ap[1] - 2)
                        else:
                            states[-1].p2 = (ap[0], ap[1] - 2)
                    #L jump Down and to the side
                    if (ap[0], ap[1] + 1) == ip and self.wall_adj(ap[0], ap[1]+1, _Direction.Down) and not (ip[1] == 0 or ip[1] == 8):
                        #Down and Right
                        if not ip[0] == 8 and not self.wall_adj(ap[0], ap[1]+1, _Direction.RIGHT):
                            states.append(self.deepcopy())
                            if p1_turn:
                                states[-1].p1 = (ap[0] + 1, ap[1] - 1)
                            else:
                                states[-1].p2 = (ap[0] + 1, ap[1] - 1)
                        #Down and Left
                        if not ip[0] == 0 and not self.wall_adj(ap[0], ap[1]+1, _Direction.LEFT):
                            states.append(self.deepcopy())
                            if p1_turn:
                                states[-1].p1 = (ap[0] - 1, ap[1] - 1)
                            else:
                                states[-1].p2 = (ap[0] - 1, ap[1] - 1)
        # Moving right
        if ap[1] < 8:
            if not self.wall_adj(ap[0], ap[1], _Direction.RIGHT):
                if (ap[0] + 1, ap[1]) != ip:
                    # Add state
                    states.append(self.deepcopy())
                    if p1_turn:
                        states[-1].p1 = (ap[0] + 1, ap[1])
                    else:
                        states[-1].p2 = (ap[0] + 1, ap[1])
                else:
                    #Right jump
                    if (ap[0] + 1, ap[1]) == ip and not self.wall_adj(ap[0] + 1, ap[1], _Direction.RIGHT) and not (ip[0] == 0 or ip[0] == 8):
                        states.append(self.deepcopy())
                        if p1_turn:
                            states[-1].p1 = (ap[0] + 2, ap[1])
                        else:
                            states[-1].p2 = (ap[0] + 2, ap[1])
            #L jump Right and veritically
                    #if ip is to the right and there is a wall to the right of ip
                    if (ap[0] + 1, ap[1]) == ip and self.wall_adj(ap[0] + 1, ap[1], _Direction.RIGHT) and not (ip[0] == 0 or ip[0] == 8):
                        #Right and Up
                        if not ip[1] == 8 and not self.wall_adj(ap[0] + 1, ap[1], _Direction.UP):
                            states.append(self.deepcopy())
                            if p1_turn:
                                states[-1].p1 = (ap[0] + 1, ap[1] + 1)
                            else:
                                states[-1].p2 = (ap[0] + 1, ap[1] + 1)
                        #Right and Down
                        if not ip[1] == 0 and not self.wall_adj(ap[0] + 1, ap[1], _Direction.DOWN):
                            states.append(self.deepcopy())
                            if p1_turn:
                                states[-1].p1 = (ap[0] + 1, ap[1] - 1)
                            else:
                                states[-1].p2 = (ap[0] + 1, ap[1] - 1)
        # Moving left
        if ap[1] < 8:
            if not self.wall_adj(ap[0], ap[1], _Direction.LEFT):
                if (ap[0] + 1, ap[1]) != ip:
                    # Add state
                    states.append(self.deepcopy())
                    if p1_turn:
                        states[-1].p1 = (ap[0] + 1, ap[1])
                    else:
                        states[-1].p2 = (ap[0] + 1, ap[1])
                else:
                    #Left jump
                    if (ap[0] - 1, ap[1]) == ip and not self.wall_adj(ap[0] - 1, ap[1], _Direction.LEFT) and not (ip[0] == 0 or ip[0] == 8):
                        states.append(self.deepcopy())
                        if p1_turn:
                            states[-1].p1 = (ap[0] - 2, ap[1])
                        else:
                            states[-1].p2 = (ap[0] - 2, ap[1])
                    #if ip is to the left and there is a wall to the left of ip
                    if (ap[0] + 1, ap[1]) == ip and self.wall_adj(ap[0] + 1, ap[1], _Direction.LEFT) and not (ip[0] == 0 or ip[0] == 8):
                        #Left and Up
                        if not ip[1] == 8 and not self.wall_adj(ap[0] + 1, ap[1], _Direction.UP):
                            states.append(self.deepcopy())
                            if p1_turn:
                                states[-1].p1 = (ap[0] - 1, ap[1] + 1)
                            else:
                                states[-1].p2 = (ap[0] - 1, ap[1] + 1)
                        #Left and Down
                        if not ip[1] == 0 and not self.wall_adj(ap[0] + 1, ap[1], _Direction.DOWN):
                            states.append(self.deepcopy())
                            if p1_turn:
                                states[-1].p1 = (ap[0] - 1, ap[1] - 1)
                            else:
                                states[-1].p2 = (ap[0] - 1, ap[1] - 1)
            
        return states

    def __str__(self):
        s = ""
        for y in range(8, -1, -1):
            # Draw row of tiles
            for x in range(9):
                # Place vertical wall segment
                if x > 0:
                    if self.wall_adj(x, y, _Direction.LEFT):
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
                    if self.wall_adj(x, y, _Direction.DOWN):
                        s += '#'
                    else:
                        s += '-'
                
                # Place corner
                if x < 8 and y > 0:
                    if y > 0 and (self.walls[0][y-1][x] or self.walls[1][y-1][x]):
                        s += '#'
                    else:
                        s += '+'
            if y > 0:
                s += '\n'
        
        # Add wall counts
        s = f"{s}P1: {self.p1_walls:2}     P2: {self.p2_walls:2}"
        return s

def main():
    board = Board()
    print(board)

if __name__ == "__main__":
    main()

