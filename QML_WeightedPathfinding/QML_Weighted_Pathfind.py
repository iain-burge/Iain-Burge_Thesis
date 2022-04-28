
from numpy import negative
from QuantumCircuits import QuantumCircuits as quantCircuits
from UtilityFunctions import QuantPrint as qPrint

from Grid import Grid

import pennylane as qml
from pennylane import numpy as np

import math

from typing import List


dimx = 2
dimy = 2

# weights = {i: ((i%4)/4 + (i//4)/4)/2 if (i%4 > 2) else -0.5 for i in range(16)}
# weightsArr = [
#   +0.0, -1.0, +0.0, +1.0, +0.0, +0.0, +0.0, +0.0,
#   +0.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0,
#   -1.0, +1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0,
#   +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0,
#   +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0,
#   +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0,
#   +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0,
#   +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0,
# ]
weightsArr = [
  +0.0, +1.0, +0.0, +0.5,
  +0.0, +0.0, +0.0, +0.0,
  +0.0, +0.0, +0.0, +0.0,
  +0.0, +0.0, +0.0, +0.0
]
# weightsArr = [
#   +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0,
#   +1.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0,
#   +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0,
#   +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0
# ]
# weightsArr = [
#   +0.0, +0.0, +0.0, +0.0,
#   +0.0, +0.0, +0.0, +0.0
# ]
weights = {i: weightsArr[i] for i in range(1<<(dimx+dimy))}


invalidStates = set([1, 2, 4])

qc = quantCircuits(
  dimx,    dimy, 
  2,       2, 
  weights, invalidStates, 
  1
)

@qml.qnode(qc.device())
def wgGridSearch(state: int=0, numIterations: int=0) -> List[float]:
  # weighted grover grid search
  #INITIALIZATION
  qml.BasisState(
    # Bitwise representation of state in np array
    np.array(
      [(state & (1<<i))>>i for i in range(qc.getStateBits()-1,-1,-1)]
      + [0 for _ in range(qc.getOutputBits())]
    ),
    # State bit wires
    wires=qc.getStateWires() + qc.getOutputWires()
  )

  # Action Wires Init
  for w in qc.getActionWires():
    qml.Hadamard(wires=w)

  #SEARCH
  for _ in range(numIterations):
    qc.oracle()
    qml.templates.GroverOperator(
      wires=qc.getActionWires()
    )

  return qml.probs(wires=qc.getActionWires())

if __name__ == "__main__":
  grid = Grid(1<<dimx, 1<<dimy, weights, set())
  grid.showGrid()

  superPath = wgGridSearch(0, 1)
  print(f"mean={np.mean([abs(el) for el in superPath])}")
  print(qPrint.stateVec2String(superPath, precision=4, meanThreshold=False, breaks=2))

  drawer = qml.draw(wgGridSearch)
  # print(drawer(0,2))
