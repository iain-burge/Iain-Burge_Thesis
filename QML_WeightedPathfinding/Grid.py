
from typing import List
import math

class Grid():
  # Static Variables
  # Constants
  RIGHT = 0
  UP    = 1
  LEFT  = 2
  DOWN  = 3

  # Grid entry subclass start
  class Entry():
    def __init__(self, state: int, location: tuple[int, int], isInvalid: bool,
        weight: int=0):
      self.state     = state
      self.location  = location
      self.isInvalid = isInvalid
      self.weight    = weight

    def display(self) -> str:
      if self.isInvalid:
        return "nval"
      return f"{self.weight:+.1f}"

    def __str__(self) -> str:
      return f"{self.state:>08b}"
    
    def __eq__(self, other) -> bool:
      return self.state == other.state
  # Grid entry subclass end

  def __init__(self, sizeX: int, sizeY: int, weights: dict[int, float],
      invalidStates: set[int]):
    self.__sizeX = sizeX
    self.__sizeY = sizeY
    self.__size  = sizeX * sizeY

    self.__grid  = []
    for y in range(sizeY):
      self.__grid.append([])
      for x in range(sizeX):
        self.__grid[-1].append(self.Entry(
          y*sizeX + x,
          (x,y),
          y*sizeX + x in invalidStates,
          0 if not y*sizeX + x in weights else weights[y*sizeX + x]
        ))
        # self.__grid.append(
        #   [self.Entry(i*sizeX+j, (j,i), (i*sizeX+j) in goals, (i*sizeX+j) in fails) for j in range(sizeX)]
        # )
    # for i in range(sizeY):
    #   self.__grid.append(
    #     [self.Entry(i*sizeX+j, (j,i), (i*sizeX+j) in goals, (i*sizeX+j) in fails) for j in range(sizeX)]
    #   )

    self.__weights       = weights
    self.__invalidStates = invalidStates

  def getPoint(self, x: int, y: int) -> Entry:
    return self.__grid[y][x]

  def getState(self, state: int) -> Entry:
    return self.getPoint(state%self.__sizeX, state//self.__sizeY)

  def constructPath(self, start:int, moves: List[int]):
    if start < 0 or start >= self.__sizeX * self.__sizeY:
      raise ValueError(f"start ({start}) out of range")

    path = [start]

    for move in moves:
      if move < 0 or move >= 4:
        raise ValueError(f"move ({move}) out of range")

      if move == self.RIGHT:
        path.append(
          ((path[-1]+1) % self.__sizeX)                   #x
          + self.__sizeX * (path[-1] // self.__sizeX)     #y
        )
      if move == self.UP:
        path.append(
          ((path[-1]) % self.__sizeX)                     #x
          + self.__sizeX * (((path[-1] // self.__sizeX)-1)%self.__sizeY)  #y
        )
      if move == self.LEFT:
        path.append(
          ((path[-1]-1) % self.__sizeX)                   #x
          + self.__sizeX * (path[-1] // self.__sizeX)     #y
        )
      if move == self.DOWN:
        path.append(
          ((path[-1]) % self.__sizeX)                     #x
          + self.__sizeX * (((path[-1] // self.__sizeX)+1)%self.__sizeY)  #y
        )
      
    return path

  @staticmethod
  def intToMoveList(moves: int, numMoves: int, moveMask: int = 0b1) -> List[int]:
    moveList = []
    # print(math.floor(math.log(moveMask, 2)) + 1)
    for _ in range(numMoves):
      # print(f"moves={moves:>04b}")
      moveList.append(moves&moveMask)
      moves >>= math.floor(math.log(moveMask, 2)) + 1
    return moveList
  
  def validPath(self, path: List[int]) -> bool:
    for i in range(len(path)-1):
      curLoc  = self.getState(path[i  ]).location
      nextLoc = self.getState(path[i+1]).location

      distX = abs(curLoc[0] - nextLoc[0])
      distY = abs(curLoc[1] - nextLoc[1])
      
      # print(f"distX:{distX}\t|\tdistX%sizeX:{distX%self.__sizeX}\t|\tdistY:{distY}\t|\tdistY%sizeY:{distY%self.__sizeY}")
      if (     not (distX%self.__sizeX==1 and distY%self.__sizeY==0))\
          and (not (distX%self.__sizeX==0 and distY%self.__sizeY==1))\
          and (not (distX%self.__sizeX==self.__sizeX-1 and distY%self.__sizeY==0)
          and (not (distX%self.__sizeX==0 and distY%self.__sizeY==self.__sizeY-1))):
        return False

    for step in path:
      if step in self.__invalidStates:
        return False
    
    return True

  def displayPath(self, path: List[int]):
    if(len(path) <= 0):
      raise ValueError("Invalid Path")

    start   = path[ 0]
    end     = path[-1]
    pathSet = set(path)

    for x in range(self.__sizeY):
      for y in range(self.__sizeX):
        print("   ", end="")
        entry = self.__grid[x][y]
        if entry.state == start:
          print("s", end="")
        elif entry.state == end:
          print("e", end="")
        elif entry.state in pathSet:
          print(">", end="")
        else:
          print(" ", end="")
        print(" ", end="")

        print(entry.display(), end="")
      print()

  def isPathPossibleHeuristic(self, state: int, numMoves: int, numActionBits: int, mask: int):
    """Depreciated"""
    for i in range(1<<(numMoves*numActionBits)):
      if self.successfulPath(self.constructPath(state, self.intToMoveList(i, numMoves, mask))):
        return True
    return False
  
  def showGridAsBin(self):
    for row in self.__grid:
      for el in row:
        print((str)(el)+" ",end="")
      print()

  def showGrid(self):
    for row in self.__grid:
      for el in row:
        print(el.display()+" ",end="")
      print()
  
  def pathValue(self, path: List[int]) -> float:
    """Returns path value if path is valid, else returns None
    Args:
        path (List[int]): sequence of locations
    """
    if not self.validPath(path):
      return None

    value = 0

    for step in path:
      value += self.getState(step).weight

    return value
      

if __name__ == "__main__":
  xSize = 4
  ySize = 4

  weights = {}
  for w in range(xSize*ySize):
    weights[w] = 1-w/10

  grid = Grid(4, 4, weights, set([0,5]))
  # grid.showGrid()

  grid.displayPath([1, 2, 3, 7, 11, 15, 14])
  print(f"Total={grid.pathValue([1, 2, 3, 7, 11, 15, 14]):.2f}")
