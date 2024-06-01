import std/[algorithm, bitops, sequtils, math]
when defined(trueRandom):
  import std/sysrand
else:
  import std/random

proc getRequiredBits(len: Natural): uint32 {.inline.} =
  if len == 0:
    result = 0'u32
  else:
    result = fastLog2(len).uint32 + 1'u32

const
  Rounds = 10
  Len = 23
  BitWidth = getRequiredBits(Len) # Bit width of the input
  BitMask = (1'u32 shl BitWidth) - 1'u32

proc genKeySet(keys: var openarray[uint32]) =
  # Generate a cryptographically secure random key set
  const size = ceilDiv(BitWidth, 8) # Calculate the number of bytes needed for the width
  for i in 0..(Rounds div 2):
    var value: uint32 = 0
    when defined(trueRandom):
      let bytes = urandom(size)
      for j in 0..bytes.high:
        value = (value shl 8) or uint32(bytes[j])
    else:
      value = rand(uint32)
    keys[i] = masked(value, BitMask)
    # keys[i] = value

var
  KeySet: array[Rounds div 2 + 1, uint32]

proc rotateLeft(x, bits, BitWidth: uint32): uint32 {.inline.} =
  (x shl bits) or (x shr (BitWidth - bits))

proc rotateRight(x, bits, BitWidth: uint32): uint32 {.inline.} =
  (x shr bits) or (x shl (BitWidth - bits))

proc arrhrEncrypt(x: uint32): uint32 =
  result = x
  for i in 0..<(Rounds div 2):
    result = (result + KeySet[i]) and BitMask
    result = rotateLeft(result, 1, BitWidth)
  result = (result + KeySet[Rounds div 2]) and BitMask

proc arrhrDecrypt(y: uint32): uint32 =
  result = (y - KeySet[Rounds div 2]) and BitMask
  for i in countdown(Rounds div 2 - 1, 0):
    result = rotateRight(result, 1, BitWidth)
    result = (result - KeySet[i]) and BitMask

proc shuffle[T](x: var openArray[T]) =
  var k = 0
  for i in 0..<nextPowerOfTwo(x.len):
    let j = arrhrEncrypt(i.uint32).int
    if j < x.len:
      assert i == arrhrDecrypt(j.uint32).int, "roundtrip failure"
      x[k] = j
      inc k

# Example usage
proc main() =
  when not defined(trueRandom):
    randomize(123)

  var data: array[Len, int]
  # var data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  for _ in 1..1:
    fill(data, 0)
    genKeySet(KeySet)
    shuffle(data)
    echo data

main()
