import std/[bitops, strutils]

# Rotate left function
proc rotateLeft(x, bits, width: uint32): uint32 =
  ((x shl bits) and ((1'u32 shl width) - 1'u32)) or (x shr (width - bits))

# ARRHR function
proc arrhr(x: uint32, keySet: openArray[uint32], rounds: int): uint32 =
  var t = x
  let width = x.countLeadingZeroBits.uint32 - 1 # Calculate width as the bit length of x
  echo width
  for i in 0..<(rounds div 2):
    t = (t + keySet[i mod keySet.len]) and ((1'u32 shl width) - 1)
    t = rotateLeft(t, 1, width)
  let y = (t + keySet[(rounds div 2) mod keySet.len]) and ((1'u32 shl width) - 1)
  return y

# Example usage
let x: uint32 = 0b101010  # some input integer
let keySet = [0b001'u32, 0b010'u32, 0b011'u32]  # example key set
let rounds = 4  # example number of rounds (even number)

let result = arrhr(x, keySet, rounds)
echo toBin(result.int, 6)
