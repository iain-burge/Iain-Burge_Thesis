
# from pennylane import numpy as np
import numpy as np
import math

from typing import List
from typing import Callable

#START OF GATES CLASS
class Gates():
  def __init__(self):
    pass

  @staticmethod
  def generateLogicGate(
        logicalExpression: Callable[[List[bool]], bool], numArgs: int
      ) -> List[List[float]]:
    """This gate takes an arbitrary logical expression L and converts it
      into a unitary matrix representation U. U takes a quantum basis vector
      |k> and an output bit |x> and outputs: |k>|x+L(k)mod 2>

    Args:
        logicalExpression (Callable[[List[bool]], bool]): Arbitrary logical
          Expression
        numArgs (int): Number of arguments in logical expression

    Returns:
        List[List[float]]: The unitary representation of logical expression
    """
    inQuantStateSize  = 1<<numArgs
    outQuantStateSize = 1<<1

    totalQuantStateSize = inQuantStateSize * outQuantStateSize

    outBit = numArgs

    permMatrix  = np.zeros(
      shape=(totalQuantStateSize, totalQuantStateSize),
      dtype=np.int8
    )
    logicVector = np.zeros(
      shape=numArgs,
      dtype=np.bool8
    )

    #Each column is what any basis vector will become
    for columnIndex in range(totalQuantStateSize):
      #Start with identity transformation for this column basis
      rowIndex = columnIndex
      
      #Convert the column index into a logical statement 
      #Were leftmost bit is the output bit
      for bitIndex in range(numArgs):
        logicVector[bitIndex] = (bool)((columnIndex & (1<<(bitIndex+1)))>>(bitIndex+1))

      #See if the logical expression is true at this value
      if logicalExpression(logicVector):
        #The logical expression is true, thus the output bit
        #of the row index should be flipped.
        rowIndex ^= 1
      
      permMatrix[rowIndex, columnIndex] = 1

    # print(permMatrix) #temp
    return permMatrix

  @staticmethod
  def generateLogicReflectionGate(
        logicalExpression: Callable[[List[bool]], bool], numArgs: int
      ) -> List[List[float]]:
    """This gate takes an arbitrary logical expression L and converts it
      into a unitary matrix representation U. U takes a quantum basis vector
      |k> and an output bit |x> and outputs: |k>|-x>

    Args:
        logicalExpression (Callable[[List[bool]], bool]): Arbitrary logical
          Expression
        numArgs (int): Number of arguments in logical expression

    Returns:
        List[List[float]]: The unitary representation of logical expression
    """
    inQuantStateSize  = 1<<numArgs

    outBit = numArgs

    permMatrix  = np.zeros(
      shape=(inQuantStateSize, inQuantStateSize),
      dtype=np.int8
    )
    logicVector = np.zeros(
      shape=numArgs,
      dtype=np.bool8
    )

    #Each column is what any basis vector will become
    for columnIndex in range(inQuantStateSize):
      #Start with identity transformation for this column basis
      rowIndex = columnIndex
      
      #Convert the column index into a logical statement 
      #Were leftmost bit is the output bit
      for bitIndex in range(numArgs):
        logicVector[bitIndex] = (bool)((columnIndex & (1<<(bitIndex)))>>(bitIndex))

      #See if the logical expression is true at this value
      if logicalExpression(logicVector):
        #The logical expression is true,
        #row index should be reflected.
        permMatrix[rowIndex, columnIndex] = -1
      else:
        permMatrix[rowIndex, columnIndex] = 1

    # print(permMatrix) #temp
    return permMatrix

  @staticmethod
  def generateFuzzyLogicGate(
      fuzzyExpression: Callable[[List[int]], float], numArgs: int) -> List[List[float]]:
    """Takes a fuzzy expression f from the domain of {0,1} to the range of [0,1],
      and creates a unitary matrix which applies that logic to a quantum state
      wherein the last bit is the output bit and the input bits are the remaining bits
      this unitary transformation Uf can be described in the following way:

        Uf|k>|0> = {|k> tensor (sqrt(f(k))|0> + sqrt(1-f(k)|1>)

    Args:
        fuzzyExpression (Callable[[List[int]], float]): 
          Fuzzy expression f:{0,1}^numArgs -> [0,1]
        numArgs (int): number of arguments the fuzzy expression takes

    Raises:
        ValueError: if range of fuzzy expression not equal to [0,1]

    Returns:
        List[List[float]]: Unitary representation
    """

    inQuantStateSize  = 1<<numArgs
    outQuantStateSize = 1<<1

    totalQuantStateSize = inQuantStateSize * outQuantStateSize

    outBit = numArgs

    unitaryMatrix  = np.zeros(
      shape=(totalQuantStateSize, totalQuantStateSize),
      dtype=np.float64
    )
    logicVector = np.zeros(
      shape=numArgs,
      dtype=np.int8
    )

    #Each column is what any basis vector will become
    for columnIndex in range(0, totalQuantStateSize, 2):
      #Start with identity transformation for this column basis
      rowIndex = columnIndex
      
      #Convert the column index into a logical statement 
      #Were leftmost bit is the output bit
      for bitIndex in range(numArgs):
        logicVector[bitIndex] = (columnIndex & (1<<(bitIndex+1)))>>(bitIndex+1)

      #Calculate the fuzzy truth of this logical statement
      fuzzyTruth = fuzzyExpression(logicVector)

      if fuzzyTruth > 1 or fuzzyTruth < 0:
        raise ValueError(f"Fuzzy Expression Range not equal to [0,1]: fuzzyTruth={fuzzyTruth}")

      fuzzyRotation = math.asin(math.sqrt(fuzzyTruth))# * math.pi / 2)
      #fuzzyRotation = fuzzyTruth * math.pi / 2
      # print(f"fuzzyTruth:{fuzzyTruth:.2f}, fuzzyRotation:{fuzzyRotation/math.pi:.2f}pi")

      # Y rotation
      unitaryMatrix[rowIndex  , columnIndex  ] =  math.cos(fuzzyRotation)
      unitaryMatrix[rowIndex  , columnIndex+1] = -math.sin(fuzzyRotation)
      unitaryMatrix[rowIndex+1, columnIndex  ] =  math.sin(fuzzyRotation)
      unitaryMatrix[rowIndex+1, columnIndex+1] =  math.cos(fuzzyRotation)

    return unitaryMatrix
  
  @staticmethod
  def generateSetGate(indexSet: set[int], unitaryGateSize: int) -> list[list[float]]:
    """Generates a unitary gate which flips the output bit if the input is an
      element of the input set. ie, if k in indexSet |k>|x> -> |k>|x+1mod2>
      if k not in index set |k>|x> -> |k>|x>.

    Args:
        indexSet (set[int]): The set of indices of interest
        unitaryGateSize (int): The number of input bits the unitary matrix will take

    Returns:
        list[list[float]]: A unitary matrix which preforms the above transformation
    """
    inQuantStateSize  = 1<<unitaryGateSize
    outQuantStateSize = 1<<1

    totalQuantStateSize = inQuantStateSize * outQuantStateSize

    permMatrix  = np.zeros(
      shape=(totalQuantStateSize, totalQuantStateSize),
      dtype=np.int8
    )

    for columnIndex in range(totalQuantStateSize):
      rowIndex = columnIndex

      if columnIndex>>1 in indexSet:
        rowIndex ^= 1
      
      permMatrix[rowIndex, columnIndex] = 1

    # print(permMatrix) #temp
    return permMatrix
  
  @staticmethod
  def generateFuzzySetGate(indexDict: dict[int, float], 
      unitaryGateSize: int) -> List[List[float]]:
    """Takes a fuzzy set with a membership function f from indices to the range of [0,1],
      and creates a unitary matrix which applies that membership function to a quantum state
      wherein the last bit is the output (membership) bit and the input bits are the remaining bits
      this unitary transformation Uf can be described in the following way:

        Uf|k>|0> = {|k> tensor (sqrt(f(k))|0> + sqrt(1-f(k)|1>)

    Args:
        indexDict (dict[int, float]): Dictionary mapping from indices to membership value in range [0,1]
        unitaryGateSize (int): The possible number of indice bits

    Returns:
        List[List[float]]: Unitary matrix which represents fuzzy membership function
    """
    inQuantStateSize  = 1<<unitaryGateSize
    outQuantStateSize = 1<<1

    totalQuantStateSize = inQuantStateSize * outQuantStateSize

    unitaryMatrix  = np.zeros(
      shape=(totalQuantStateSize, totalQuantStateSize),
      dtype=np.float64
    )

    for columnIndex in range(0, totalQuantStateSize, 2):
      rowIndex = columnIndex

      if columnIndex>>1 in indexDict.keys():
        fuzzyRotation = math.asin(
          np.sign(indexDict[columnIndex>>1]) * math.sqrt(abs(indexDict[columnIndex>>1]))
        )
      else:
        fuzzyRotation = 0
      
      # Y rotation
      unitaryMatrix[rowIndex  , columnIndex  ] =  math.cos(fuzzyRotation)
      unitaryMatrix[rowIndex  , columnIndex+1] = -math.sin(fuzzyRotation)
      unitaryMatrix[rowIndex+1, columnIndex  ] =  math.sin(fuzzyRotation)
      unitaryMatrix[rowIndex+1, columnIndex+1] =  math.cos(fuzzyRotation)

    return unitaryMatrix

  @staticmethod
  def grid2DTraversalGate(direction: int, numXBits: int, numYBits: int) -> List[List[float]]:
    """Let each basis quantum state represent a position in a grid, this function
      Creates a unitary matrix which transforms a state corresponding to the
      input direction. For example, if direction=0 then the unitary matrix will
      change the state such that the new state is one position to the right.

    Args:
        direction (int): Direction this unitary matrix will represent, in range [0,3]
          0: Right, 1: Up, 2: Left, 3: Down
        numXBits (int):  The number of bits representing the X axis
        numYBits (int):  The number of bits representing the Y axis

    Raises:
        ValueError: Error when direction is not in the range [0,3]

    Returns:
        List[List[float]]: Unitary matrix described above
    """
    if direction not in range(4):
      raise ValueError('direction must be int in range [0,3]')

    quantumStateSize = 1<<(numYBits + numXBits)

    permMatrix  = np.zeros(
      shape=(quantumStateSize, quantumStateSize),
      dtype=np.int8
    )

    xBitMask = (1<<(numXBits)) - 1
    yBitMask = ~xBitMask
    #print("xBitMask",xBitMask)

    if(direction == 0):
      #direction is right
      nextState = lambda state:(
        (((state & xBitMask) + 1) % (1<<numXBits))  #x bits
        + (state & yBitMask)                        #y bits
      )
    if(direction == 1):
      #direction is up
      nextState = lambda state: (
        (state & xBitMask)                                        #x bits
        + ((((state>>numXBits) - 1) % (1<<numYBits)) << numXBits) #y bits
      )
    if(direction == 2):
      #direction is left
      nextState = lambda state:(
        (((state & xBitMask) - 1) % (1<<numXBits))  #x bits
        + (state & yBitMask)                        #y bits
      )
    if(direction == 3):
      #direction is down
      nextState = lambda state:(
        (state & xBitMask)                                        #x bits
        + ((((state>>numXBits) + 1) % (1<<numYBits)) << numXBits) #y bits
      )
    
    # print(f"direction: {direction}")
    # for state in range(quantumStateSize):
      # print(f"state: {state:>04b}\tnextState {nextState(state):>04b}")

    for columnIndex in range(quantumStateSize):
      rowIndex = nextState(columnIndex)

      permMatrix[rowIndex, columnIndex] = 1
    
    # print(permMatrix)
    return permMatrix

  @staticmethod
  def generateCountGate(numBits: int) -> List[List[float]]:
    """ This Gate Does the following operation on a sequence of qubits
      in the standard basis:
        |k> -> |k+1 mod(1<<numBits)>

    Args:
        numBits (int):  The number of bits representing the count

    Returns:
        List[List[float]]: Unitary matrix described above
    """

    quantumStateSize = 1<<(numBits)

    permMatrix  = np.zeros(
      shape=(quantumStateSize, quantumStateSize),
      dtype=np.int8
    )

    nextState = lambda state:(
      (state+1) % (1<<numBits)
    )

    for columnIndex in range(quantumStateSize):
      rowIndex = nextState(columnIndex)

      permMatrix[rowIndex, columnIndex] = 1
    
    # print(permMatrix)
    return permMatrix
#END OF GATES CLASS
    

if __name__ == "__main__":
  print(3*"\n")
  gates = Gates()
  # gates.generateLogicGate((lambda premises: premises[0] and premises[1]), 2)
  # gates.generateLogicGate(lambda premises: premises[0], 1)
  # gates.generateLogicGate(
  #   lambda premises: (premises[0] and premises[1]) or (premises[0] and premises[2]), 3
  # )
  # gates.generateSetGate(set([0,3]), 2)
  print(gates.grid2DTraversalGate(0,2,2))
  print(gates.generateLogicGate(lambda p: not(p[0] and p[1]), 2))
  print(gates.generateLogicReflectionGate(lambda p: p[0] and not p[1], 2))

  #Fuzzy Gate test
  print("Fuzzy Logic Gate")
  fuzzyLogic = gates.generateFuzzyLogicGate(lambda p: (2/3)*p[0] + (1/3)*p[1], 2)
  np.set_printoptions(precision=2)
  print("Pure:\n"+np.array_str(fuzzyLogic, precision=2, suppress_small=True))
  print("Squared:\n"+np.array_str(np.square(fuzzyLogic), precision=2, suppress_small=True))
  
  print("x and not y gate test:")
  andNotTest = Gates.generateFuzzyLogicGate(lambda args: args[0] * (1 - args[1]), 2)
  print(np.array_str(andNotTest, precision=2, suppress_small=True))

  print("\n\nFuzzy Set Gate")
  fuzzySet = gates.generateFuzzySetGate({1: 0.6, 1: -0.5}, 1)
  np.set_printoptions(precision=2)
  print("Pure:\n"+np.array_str(fuzzySet, precision=2, suppress_small=True))
  print("Squared:\n"+np.array_str(np.square(fuzzySet), precision=2, suppress_small=True))

  print("\n\nCount Gate")
  countGate = gates.generateCountGate(4)
  print(countGate)
