
import math
from telnetlib import GA
from tkinter import W
from UnitaryGates import Gates
from Grid import Grid

import pennylane as qml
from pennylane import numpy as np

from Contants import DEBUG

from typing import List

class QuantumCircuits():
  POSITIVE = 0
  NEGATIVE = 1

  def __init__(self, xbits: int, ybits: int, numActions: int, 
      numUnitaryActionBits: int, weights: dict[int, float], 
      invalidStates: set[int], decayRate: float):
    #Environment
    self.__xbits = xbits
    self.__ybits = ybits

    #Actions
    self.__numActions           = numActions
    self.__numUnitaryActionBits = numUnitaryActionBits
    self.__unitaryActions       = self.__initUnitaryActionList()

    #Bit Distribution
    self.__actionBits      = numActions * numUnitaryActionBits
    self.__stateBits       = xbits + ybits
    self.__rewardBits      = 2
    self.__finalRewardBits = 1
    self.__invalidBits     = math.floor(math.log2(numActions)) + 1
    self.__auxInvalidBits  = 1
    self.__outputBits      = 1

    self.__totalQBits = self.__actionBits + self.__stateBits\
      + self.__rewardBits + self.__finalRewardBits + self.__invalidBits\
      + self.__auxInvalidBits + self.__outputBits

    #Wire Naming
    wireNum = 0

    # actions wires stored in a matrix where the rows represent action number
    # and the columns represent the wires representing the action at that stage
    self.__actionWiresMatrix = []
    self.__actionWires       = []
    for action in range(numActions):
      self.__actionWiresMatrix.append([])
      for unitaryActionBit in range(numUnitaryActionBits):
        self.__actionWiresMatrix[-1].append(wireNum)
        self.__actionWires.append(wireNum)
        wireNum += 1

    self.__stateWires       = [wireNum + i for i in range(self.__stateBits)]
    wireNum                += self.__stateBits
    self.__rewardWires      = [wireNum + i for i in range(self.__rewardBits)]
    wireNum                += self.__rewardBits
    self.__finalRewardWires = [wireNum + i for i in range(self.__finalRewardBits)]
    wireNum                += self.__finalRewardBits
    self.__invalidWires     = [wireNum + i for i in range(self.__invalidBits)]
    wireNum                += self.__invalidBits
    self.__auxInvalidWires  = [wireNum + i for i in range(self.__auxInvalidBits)]
    wireNum                += self.__auxInvalidBits
    self.__outputWires      = [wireNum + i for i in range(self.__outputBits)]
    wireNum                += self.__stateBits

    self.__wireNames = self.__actionWires + self.__stateWires\
      + self.__rewardWires + self.__finalRewardWires + self.__invalidWires\
      + self.__auxInvalidWires + self.__outputWires 

    #Reinforcement Leaning
    self.__invalidStates    = invalidStates
    self.__weights          = weights
    self.__positiveWeights  =\
      {key: max(self.__weights[key], 0) for key in weights.keys()}
    self.__negativeWeights  =\
      {key: min(self.__weights[key], 0) for key in weights.keys()}

    self.__discountFactors = self.__initDiscountFactors(decayRate, numActions)

    self.__unitaryWeightsArrayPos = []
    self.__unitaryWeightsArrayNeg = []
    self.__unitaryWeightsArray    = []
    for action in range(numActions):
      discountedWeightsPos = self.__positiveWeights.copy()
      discountedWeightsNeg = self.__negativeWeights.copy()
      discountedWeights    = {key : self.__weights[key]/2 for key in weights.keys()}
      for key in discountedWeightsPos.keys():
        discountedWeightsPos[key] *= self.__discountFactors[action]
      for key in discountedWeightsNeg.keys():
        discountedWeightsNeg[key] *= self.__discountFactors[action]
      for key in discountedWeights.keys():
        discountedWeights[key]    *= self.__discountFactors[action]

      self.__unitaryWeightsArrayPos.append(np.array(
        Gates.generateFuzzySetGate(
          discountedWeightsPos, self.__stateBits
      )))
      self.__unitaryWeightsArrayNeg.append(np.array(
        Gates.generateFuzzySetGate(
          discountedWeightsNeg, self.__stateBits
      )))
      self.__unitaryWeightsArray.append(np.array(
        Gates.generateFuzzySetGate(
          discountedWeights,    self.__stateBits
      )))
    
    self.__unitaryCalculateOutputReward = Gates.generateFuzzyLogicGate(
      # positive and not negative
      lambda p: p[self.POSITIVE] * (1-p[self.NEGATIVE]), 2
    )

    #Invalid States
    self.__invalidStates = invalidStates
    self.__unitaryInvalidStates = Gates.generateSetGate(
      invalidStates, self.__stateBits
    )

    self.__unitaryCount = Gates.generateCountGate(self.__invalidBits)
    self.__unitaryOr    = Gates.generateFuzzyLogicGate(
      lambda p: sum(p) != 0, self.__invalidBits
    )
    self.__calcOutput   = Gates.generateLogicGate(
      lambda p: p[0] and not p[1], 2
    )

    #Init Quantum Device
    self.__dev = qml.device('default.qubit', wires=self.__wireNames)
  
  #getter functions
  def getActionBits(self):
    return self.__actionBits
  def getStateBits(self):
    return self.__stateBits
  def getRewardBits(self):
    return self.__rewardBits
  def getOutputBits(self):
    return self.__outputBits
  def getActionWires(self):
    return self.__actionWires
  def getStateWires(self):
    return self.__stateWires
  def getRewardWires(self):
    return self.__rewardWires
  def getFinalRewardWires(self):
    return self.__finalRewardWires
  def getInvalidWires(self):
    return self.__invalidWires
  def getAuxInvalidWires(self):
    return self.__auxInvalidWires
  def getOutputWires(self):
    return self.__outputWires

  def __initUnitaryActionList(self) -> List[List[List[float]]]:
    """List of unitary matrices which represent actions of grid navigation
    """
    return [
      #Move Right
      np.array(
        Gates.grid2DTraversalGate(Grid.RIGHT, self.__xbits, self.__ybits)
      ),
      #Move Up
      np.array(
        Gates.grid2DTraversalGate(Grid.UP,    self.__xbits, self.__ybits)
      ),
      #Move Left
      np.array(
        Gates.grid2DTraversalGate(Grid.LEFT,  self.__xbits, self.__ybits)
      ),
      #Move Down
      np.array(
        Gates.grid2DTraversalGate(Grid.DOWN,  self.__xbits, self.__ybits)
      )
    ]
  
  @staticmethod
  def __initDiscountFactors(decayRate: float, numActions: int):
    discountFactors = []
    total           = 0

    for i in range(numActions):
      total += decayRate**i
    for i in range(numActions):
      discountFactors.append((decayRate**i)/total)

    return discountFactors

  def preformAction(self, actionNum: int) -> None:
    for unitaryActionInd in range(len(self.__unitaryActions)):
      for bit in range(self.__numUnitaryActionBits):
        if unitaryActionInd%(1<<(self.__numUnitaryActionBits-bit-1)) == 0:
          wireInd = bit + (self.__numUnitaryActionBits * actionNum)
          qml.PauliX(wires=self.__actionWires[wireInd])
      qml.ControlledQubitUnitary(
        self.__unitaryActions[unitaryActionInd],           #Unitary Matrix
        control_wires=self.__actionWiresMatrix[actionNum], #Control Wires
        wires=self.__stateWires                            #In/Out  Wires
      )

  def preformActionInverse(self, actionNum: int) -> None:
    for unitaryActionInd in range(len(self.__unitaryActions)):
      for bit in range(self.__numUnitaryActionBits):
        if unitaryActionInd%(1<<(self.__numUnitaryActionBits-bit-1)) == 0:
          wireInd = bit + (self.__numUnitaryActionBits * actionNum)
          qml.PauliX(wires=self.__actionWires[wireInd])
      qml.ControlledQubitUnitary(
        self.__unitaryActions[unitaryActionInd],           #Unitary Matrix
        control_wires=self.__actionWiresMatrix[actionNum], #Control Wires
        wires=self.__stateWires                            #In/Out  Wires
      ).inv()

  def checkValidity(self):
    qml.QubitUnitary(
      self.__unitaryInvalidStates,
      wires=self.__stateWires + self.__auxInvalidWires
    )
    qml.ControlledQubitUnitary(
      self.__unitaryCount,
      control_wires=self.__auxInvalidWires,
      wires=self.__invalidWires
    )
    qml.QubitUnitary(
      self.__unitaryInvalidStates,
      wires=self.__stateWires + self.__auxInvalidWires
    ).inv()

  def checkValidityInverse(self):
    qml.QubitUnitary(
      self.__unitaryInvalidStates,
      wires=self.__stateWires + self.__auxInvalidWires
    )
    qml.ControlledQubitUnitary(
      self.__unitaryCount,
      control_wires=self.__auxInvalidWires,
      wires=self.__invalidWires
    ).inv()
    qml.QubitUnitary(
      self.__unitaryInvalidStates,
      wires=self.__stateWires + self.__auxInvalidWires
    ).inv()
  
  def evaluateState(self, actionNum):
    #Updates output bit rotation according to state value
    #Positive rotation:
    qml.QubitUnitary(
      self.__unitaryWeightsArrayPos[actionNum],
      wires= self.__stateWires + [self.__rewardWires[self.POSITIVE]]
    )
    #Negative rotation:
    qml.QubitUnitary(
      self.__unitaryWeightsArrayNeg[actionNum],
      wires= self.__stateWires + [self.__rewardWires[self.NEGATIVE]]
    )

  def evaluateStateInverse(self, actionNum):
    #Negative inverse rotation:
    qml.QubitUnitary(
      self.__unitaryWeightsArrayNeg[actionNum],
      wires= self.__stateWires + [self.__rewardWires[self.NEGATIVE]]
    ).inv()
    #Positive inverse rotation:
    qml.QubitUnitary(
      self.__unitaryWeightsArrayPos[actionNum],
      wires= self.__stateWires + [self.__rewardWires[self.POSITIVE]]
    ).inv()
  
  def updateOutput(self):
    # Rewards
    qml.QubitUnitary(
      self.__unitaryCalculateOutputReward,
      wires= self.__rewardWires[::-1] + self.__finalRewardWires
    )
    #Invalids
    qml.QubitUnitary(
      self.__unitaryOr,
      wires=self.__invalidWires + self.__auxInvalidWires
    )
    #Output
    qml.QubitUnitary(
        self.__calcOutput,
        wires= self.__auxInvalidWires + self.__finalRewardWires\
          + self.__outputWires
    )

  def updateOutputInverse(self):
    #Output
    qml.QubitUnitary(
        self.__calcOutput,
        wires= self.__auxInvalidWires + self.__finalRewardWires\
          + self.__outputWires
    ).inv()
    #Invalids
    qml.QubitUnitary(
      self.__unitaryOr,
      wires=self.__invalidWires + self.__auxInvalidWires
    ).inv()
    #Rewards
    qml.QubitUnitary(
      self.__unitaryCalculateOutputReward,
      wires= self.__rewardWires[::-1] + self.__finalRewardWires
    ).inv()

  def oracle(self) -> None:
    for action in range(self.__numActions):
      self.preformAction(action)
      self.evaluateState(action)
      self.checkValidity()
    self.updateOutput()
    
    # return
    qml.PauliZ(self.__outputWires)

    self.updateOutputInverse()
    for action in range(self.__numActions-1,-1,-1):
      self.checkValidityInverse()
      self.evaluateStateInverse(action)
      self.preformActionInverse(action)
  
  def device(self):
    return self.__dev

  @staticmethod
  def debugUnitaryMatrix(unitaryMatrix: List[List[float]], name: str = "unitary matrix"):
    np.set_printoptions(linewidth=200)

    print(f"{name} | shape {np.shape(unitaryMatrix)}:")
    print(np.array_str(unitaryMatrix, precision=2, suppress_small=True))
 
    
if __name__ == "__main__":
  weights   = {i: i/10 - 1 for i in range(4*4)}
  qCircuits = QuantumCircuits(1,1,4,2, weights, set(), 0.75, False)
