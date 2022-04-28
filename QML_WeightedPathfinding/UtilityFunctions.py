
from typing import List
import math
import cmath
import numpy as np

class QuantPrint():
  """Utility functions for understanding pennylane's outputs.
    Not intended to be well optimized, just what I need for the moment
  """
  @staticmethod
  def intToBinStr(val: int, numBits: int, breaks: int = -1) -> str:
    """Takes an int and converts it to a binary representation

    Args:
        val (int): The numerical value
        numBits (int): The number of bits being used

    Returns:
        str: String of binary representation
    """
    out = ""
    for i in range(numBits):
      if breaks != -1 and 0<i<numBits and i%breaks==0:
        out += ","
      out += ((str)((val&(1<<(numBits-i-1)))>>(numBits-i-1)))[0]
    return out

  @staticmethod
  def stateVec2String(vector: List[complex], numBits: int = None, name: str = "psi",
      ignoreZeros: bool = True, precision: int = 2, meanThreshold: bool = False,
      breaks: int = -1) -> str:
    """Generates a string representation of a quantum state in the elementary basis.

    Args:
        vector (List[complex]):      The Quantum state in complex vector form.
        numBits (int, optional):     Number of quantum bits your state has. Defaults to None.
        name (str, optional):        Quantum state's name. Defaults to "psi".
        ignoreZeros(bool, optional): Ignores 0 amplitude bases vectors if true.
        precision(int, optional):    How precise the coefficients are.
        threshold(bool, optional):   Minimum probability to be printed is the mean

    Returns:
        str: String representation of quantum state.
    """
    if(numBits == None):
      numBits = math.ceil(math.log2(len(vector)))

    out = f"|{name}> ="
    threshold = np.mean([abs(el) for el in vector]) if meanThreshold else 0
    minval = max(10**(-precision), threshold)
    firstBit = True
    for i in range(len(vector)):
      if not ((np.abs(vector[i]) < minval) and ignoreZeros):
        if not firstBit:
          out += " +"
        else:
          firstBit = False
        val = complex(vector[i])

        if abs(val.real) > minval and abs(val.imag) > minval:
          out += f" ({val.real:.{precision}f}{val.imag:+.{precision}f}i)"
        elif abs(val.real) > minval:
          out += f" ({val.real:.{precision}f})"
        elif abs(val.imag) > minval:
          out += f" ({val.imag:.{precision}f}i)"

        out += f"|{QuantPrint.intToBinStr(i, numBits, breaks)}>"
    return out
  
  @staticmethod
  def stateVec2StringExt(vector: List[complex], wires: List[int],
      numBits: int = None, name: str = "psi") -> str:
    """This function is probably not going to work for now, so ignore it

    Args:
        vector (List[complex]): [description]
        wires (List[int]): [description]
        numBits (int, optional): [description]. Defaults to None.
        name (str, optional): [description]. Defaults to "psi".

    Raises:
        ValueError: [description]

    Returns:
        str: [description]
    """
    if(numBits == None):
      numBits = math.ceil(math.log2(len(vector)))
    
    ignoredWires = set([i for i in range(numBits)])
    for wire in wires:
      if(wire>numBits or wire<0):
        raise ValueError("Wire in wires outside of range of bits")
      ignoredWires.remove(wire)
    ignoredWires = list(ignoredWires)

    out = f"|{name}> ="
    for skeleton in range(1<<len(wires)):
      pass
    print(ignoredWires)

if __name__ == "__main__":
  # print(QuantPrint.stateVec2String([complex(.25,.25),.5,.5,.5], 4))
  print(QuantPrint.stateVec2StringExt([complex(0,.25),.5,.5,.5], [0,1], 4))
  print(QuantPrint.stateVec2String([complex(.25,-.25),.5,.5,.5], 4))