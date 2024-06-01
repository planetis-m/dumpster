import std/[bitops, sequtils]

# Define the key set (precomputed for simplicity)
const
  Rounds = 10
  BitWidth = 32'u32 # Bit width of the input
  BitMask = (1'u32 shl BitWidth) - 1'u32

  KeySet: array[10, uint32] = mapLiterals([
    0x73E9D5D1, 0xA2C5442C, 0xC9E94D8C, 0xCA8EDF64, 0x6813909B,
    0xFC008911, 0x7C2145A0, 0x307B8035, 0x88BAA1DE, 0xAF375D34
  ], uint32)

# proc rotateLeft(x, bits, BitWidth: uint32): uint32 {.inline.} =
#   (x shl bits) or (x shr (BitWidth - bits))
#
# proc rotateRight(x, bits, BitWidth: uint32): uint32 {.inline.} =
#   (x shr bits) or (x shl (BitWidth - bits))

proc arrhrEncrypt(x: uint32): uint32 =
  result = x
  for i in 0..<(Rounds div 2):
    result = (result + KeySet[i mod Rounds]) and BitMask
    result = rotateLeftBits(result, 1)
  result = (result + KeySet[Rounds div 2 mod Rounds]) and BitMask

proc arrhrDecrypt(y: uint32): uint32 =
  result = (y - KeySet[Rounds div 2 mod Rounds]) and BitMask
  for i in countdown(Rounds div 2 - 1, 0):
    result = rotateRightBits(result, 1)
    result = (result - KeySet[i mod Rounds]) and BitMask

proc shuffle[T](x: var openArray[T]) =
  for i in 0..<len(x):
    let j = arrhrEncrypt(i.uint32).int
    if j < x.len:
      assert i == arrhrDecrypt(j.uint32).int, "roundtrip failure"
      swap(x[i], x[j])

# Example usage
proc main() =
  var data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  shuffle(data)
  var first = true
  for i in 0..<data.len:
    stdout.write (if first: "" else: ", "), data[i]
    first = false
  stdout.write "\n"

main()
