import std/[bitops, sequtils, math]
when defined(trueRandom):
  import std/sysrand
else:
  import std/random

proc getRequiredBits(len: Natural): uint32 {.inline.} =
  if len == 0:
    result = 0'u32
  else:
    result = fastLog2(len).uint32 + 1'u32
    if (result and 1) != 0:
      inc(result)

const
  Rounds = 10
  Len = 128
  BitWidth = getRequiredBits(Len) # Bit width of the input
  BitMask = (1'u32 shl BitWidth) - 1'u32

proc genKeySet(keys: var openarray[uint32]) =
  # Generate a cryptographically secure random key set
  const size = ceilDiv(BitWidth, 8) # Calculate the number of bytes needed for the width
  for i in 0..<Rounds:
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
  KeySet: array[Rounds, uint32]

proc rotateLeft(x, bits, BitWidth: uint32): uint32 {.inline.} =
  (x shl bits) or (x shr (BitWidth - bits))

proc rotateRight(x, bits, BitWidth: uint32): uint32 {.inline.} =
  (x shr bits) or (x shl (BitWidth - bits))

proc arrhrEncrypt(x: uint32): uint32 =
  result = x
  for i in 0..<(Rounds div 2):
    result = (result + KeySet[i mod Rounds]) and BitMask
    result = rotateLeft(result, 1, BitWidth)
  result = (result + KeySet[Rounds div 2 mod Rounds]) and BitMask

proc arrhrDecrypt(y: uint32): uint32 =
  result = (y - KeySet[Rounds div 2 mod Rounds]) and BitMask
  for i in countdown(Rounds div 2 - 1, 0):
    result = rotateRight(result, 1, BitWidth)
    result = (result - KeySet[i mod Rounds]) and BitMask

proc shuffle[T](x: var openArray[T]) =
  for i in 0..<len(x):
    let j = arrhrEncrypt(i.uint32).int
    if j < x.len:
      assert i == arrhrDecrypt(j.uint32).int, "roundtrip failure"
      swap(x[i], x[j])

# Example usage
proc main() =
  when not defined(trueRandom):
    randomize()

  var data: array[Len, int]
  for i in 0..<Len:
    data[i] = i
  # var data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  for _ in 1..12:
    genKeySet(KeySet)
    shuffle(data)
    echo data

main()
