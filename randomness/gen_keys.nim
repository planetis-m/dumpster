import std/[sysrand, bitops, math, strutils, sugar]

const
  Rounds = 10
  BitWidth = 32'u32 # Bit width of the input
  BitMask = (1'u32 shl BitWidth) - 1'u32

var
  keys: array[Rounds, uint32]

proc genKeySet(keys: var openarray[uint32]) =
  # Generate a cryptographically secure random key set
  const size = ceilDiv(BitWidth, 8) # Calculate the number of bytes needed for the width
  for i in 0..<Rounds:
    let bytes = urandom(size)
    var value: uint32 = 0
    for j in 0..bytes.high:
      value = (value shl 8) or uint32(bytes[j])
    keys[i] = masked(value, BitMask)

proc main =
  genKeySet(keys)
  let result = collect:
    for key in keys.items:
      "0x" & key.toHex
  var first = true
  for x in result.items:
    if first:
      first = false
    else:
      stdout.write(", ")
    stdout.write(x)
  stdout.write("\n")

main()
