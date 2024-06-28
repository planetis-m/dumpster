from std/bitops import countLeadingZeroBits

proc log2Floor*(x: int): int {.inline.} =
  # Undefined for zero argument.
  assert x > 0
  result = sizeof(int)*8 - 1 - countLeadingZeroBits(x)

proc log2Ceil*(x: int): int {.inline.} =
  # Special case: if the argument is zero, returns zero.
  if x <= 1:
    result = 0
  else:
    result = sizeof(int)*8 - countLeadingZeroBits(x - 1)

proc nextPowerOfTwo*(x: int): int {.inline.} =
  # This is equivalent to pow2(log2Ceil(x)). Undefined for x<2.
  result = 1 shl (sizeof(int)*8 - countLeadingZeroBits(x - 1))

proc pow2*(power: uint): uint {.inline.} =
  # Raise 2 into the specified power.
  result = 1'u shl power

proc alignUp*(n, align: uint): uint {.inline.} =
  (n + align - 1) and (not (align - 1))

proc alignUp(p: pointer, align: uint): pointer {.inline.} =
  cast[pointer](alignUp(cast[uint](p), align))

proc alignOffset*(base: pointer, align: uint): uint {.inline.} =
  align - (cast[uint](base) and (align - 1))
