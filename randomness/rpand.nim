# Types and Structures Definition

# Global Variables Definition

var baseSeed: uint64 = 0          # SplitMix64 actual seed
var baseState: array[4, uint32]   # Xoshiro128** state, initialized by SplitMix64

# Module internal functions definition

proc splitMix64(): uint64 =
  # SplitMix64 generator info:
  #
  #   Written in 2015 by Sebastiano Vigna (vigna@acm.org)
  #
  #   To the extent possible under law, the author has dedicated all copyright
  #   and related and neighboring rights to this software to the public domain
  #   worldwide. This software is distributed without any warranty.
  #
  #   See <http://creativecommons.org/publicdomain/zero/1.0/>.
  #
  #
  #   This is a fixed-increment version of Java 8's SplittableRandom generator
  #   See http://dx.doi.org/10.1145/2714064.2660195 and
  #   http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html
  #
  #   It is a very fast generator passing BigCrush, and it can be useful if
  #   for some reason you absolutely want 64 bits of state.
  baseSeed += 0x9e3779b97f4a7c15'u64
  var z = baseSeed
  z = (z xor (z shr 30)) * 0xbf58476d1ce4e5b9'u64
  z = (z xor (z shr 27)) * 0x94d049bb133111eb'u64
  return z xor (z shr 31)

proc rotateLeft(x: uint32, k: int): uint32 {.inline.} =
  return (x shl k) or (x shr (32 - k))

proc xoshiro(): uint32 =
  # Xoshiro128** generator info:
  #
  #   Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
  #
  #   To the extent possible under law, the author has dedicated all copyright
  #   and related and neighboring rights to this software to the public domain
  #   worldwide. This software is distributed without any warranty.
  #
  #   See <http://creativecommons.org/publicdomain/zero/1.0/>.
  #
  #   This is xoshiro128** 1.1, one of our 32-bit all-purpose, rock-solid
  #   generators. It has excellent speed, a state size (128 bits) that is
  #   large enough for mild parallelism, and it passes all tests we are aware
  #   of.
  #
  #   Note that version 1.0 had mistakenly s[0] instead of s[1] as state
  #   word passed to the scrambler.
  #
  #   For generating just single-precision (i.e., 32-bit) floating-point
  #   numbers, xoshiro128+ is even faster.
  #
  #   The state must be seeded so that it is not everywhere zero.
  result = rotateLeft(baseState[1] * 5, 7) * 9
  let tmp = baseState[1] shl 9

  baseState[2] = baseState[2] xor baseState[0]
  baseState[3] = baseState[3] xor baseState[1]
  baseState[1] = baseState[1] xor baseState[2]
  baseState[0] = baseState[0] xor baseState[3]

  baseState[2] = baseState[2] xor tmp

  baseState[3] = rotateLeft(baseState[3], 11)

# Module functions definition

proc setSeed*(seed: uint64) =
  ## Set state for Xoshiro128**
  ## NOTE: We use a custom generation algorithm using SplitMix64
  baseSeed = seed # Set SplitMix64 seed for further use

  # To generate the Xoshiro128** state, we use SplitMix64 generator first
  # We generate 4 pseudo-random 64bit numbers that we combine using their LSB|MSB
  baseState[0] = cast[uint32](splitMix64() and 0xffffffff'u64)
  baseState[1] = cast[uint32]((splitMix64() and 0xffffffff00000000'u64) shr 32)
  baseState[2] = cast[uint32](splitMix64() and 0xffffffff'u64)
  baseState[3] = cast[uint32]((splitMix64() and 0xffffffff00000000'u64) shr 32)

proc getValue*(min, max: int32): int32 =
  ## Get random value within a range, min and max included
  result = cast[int32](xoshiro() mod cast[uint32](abs(max - min) + 1) + cast[uint32](min))

proc loadRandSeq*(count: Natural, min, max: int32): seq[int32] =
  # Load pseudo-random numbers with no duplicates, min and max included.
  result = @[]

  assert(count <= abs(max - min) + 1,
    "WARNING: Sequence count required is greater than range provided")

  result.setLen(count)

  var value: int32 = 0
  var valueIsDup = false

  var i = 0
  while i < count:
    value = getValue(min, max)

    for j in 0..<i:
      if result[j] == value:
        valueIsDup = true
        break

    if not valueIsDup:
      result[i] = value
      inc i

    valueIsDup = false

setSeed(124)
echo getValue(1, 4)
echo loadRandSeq(5, 1, 5)
