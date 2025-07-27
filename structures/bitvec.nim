import std/bitops

const
  BitsPerElement = sizeof(uint) * 8
  BitShift = 6 # log2(64) = 6
  BitMask = BitsPerElement - 1

type
  BitVector* = object
    capacity: int
    data: seq[uint]

proc initBitVector*(capacity: Positive = 1): BitVector =
  ## Creates a new BitVector with the specified capacity
  let numElements = (capacity + BitsPerElement - 1) shr BitShift
  result = BitVector(
    capacity: capacity,
    data: newSeq[uint](numElements)
  )

func len*(x: BitVector): int {.inline.} =
  ## Returns the capacity of the BitVector
  x.capacity

proc get*(x: BitVector, idx: int): bool =
  ## Returns true if the bit at the specified index is set, false otherwise
  rangeCheck(0 <= idx and idx < x.capacity)
  # Calculate array index and bit position within the element
  let arrayIdx = idx shr BitShift
  let bitPos = idx and BitMask
  # Create the mask for the specific bit
  let mask = 1'u64 shl bitPos
  # Use bitwise AND to check if the bit is set
  (x.data[arrayIdx] and mask) != 0

proc set*(x: var BitVector, idx: int) =
  ## Sets the bit at the specified index to 1
  rangeCheck(0 <= idx and idx < x.capacity)
  # Calculate array index and bit position within the element
  let arrayIdx = idx shr BitShift
  let bitPos = idx and BitMask
  # Create the mask for the specific bit
  let mask = 1'u64 shl bitPos
  # Use bitwise OR to set the bit
  x.data[arrayIdx] = x.data[arrayIdx] or mask

proc flip*(x: var BitVector, idx: int) =
  ## Flips the bit at the specified index (0 becomes 1, 1 becomes 0)
  rangeCheck(0 <= idx and idx < x.capacity)
  # Calculate array index and bit position within the element
  let arrayIdx = idx shr BitShift
  let bitPos = idx and BitMask
  # Create the mask for the specific bit
  let mask = 1'u64 shl bitPos
  # Use bitwise XOR to flip the bit
  x.data[arrayIdx] = x.data[arrayIdx] xor mask

func popcount*(x: BitVector): int =
  ## Returns the number of set bits (population count)
  result = 0
  for i in 0..<x.data.len:
    result += countSetBits(x.data[i])

func `==`*(a, b: BitVector): bool =
  ## Equality comparison between BitVectors
  if a.capacity != b.capacity:
    return false
  for i in 0..<a.data.len:
    if a.data[i] != b.data[i]:
      return false
  result = true

proc `$`*(x: BitVector): string =
  ## Converts the bit vector to a string representation
  if x.capacity == 0:
    return ""

  result = newString(x.capacity)
  for i in 0..<x.capacity:
    result[i] = chr(int(x.get(i)) + ord('0'))

when isMainModule:
  var x = initBitVector(5)

  x.set(1)
  x.set(2)
  echo x # Output: 01100
  echo x.get(2) # Output: true
  echo x.get(4) # Output: false
  x.flip(4)
  echo x # Output: 01101
  x.flip(4)
  echo x # Output: 01100
