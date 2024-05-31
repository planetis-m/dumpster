# https://www.ea.com/seed/news/constant-time-stateless-shuffling
type
  StatelessShuffle = object
    roundCount: uint32
    halfIndexBits: uint32
    halfIndexBitsMask: uint32
    seed: uint32 # The "random seed" that determines ordering

proc setSeed*(s: var StatelessShuffle, seed: uint32) {.inline.} =
  s.seed = seed

proc setIndexBits*(s: var StatelessShuffle, bits: uint32) {.inline.} =
  s.halfIndexBits = bits div 2'u32
  s.halfIndexBitsMask = (1'u32 shl s.halfIndexBits) - 1'u32

proc setRoundCount*(s: var StatelessShuffle, count: uint32) {.inline.} =
  s.roundCount = count

proc pcgHash(x: uint32): uint32 =
  # https://www.pcg-random.org/
  # https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
  let state = x * 747796405'u32 + 2891336453'u32
  let word = ((state shr ((state shr 28'u32) + 4'u32)) xor state) * 277803737'u32
  result = (word shr 22'u32) xor word

proc roundFunction(s: StatelessShuffle, x: uint32): uint32 =
  result = (pcgHash(x xor s.seed)) and s.halfIndexBitsMask

proc encrypt(self: StatelessShuffle, index: uint32): uint32 =
  var left = index shr self.halfIndexBits
  var right = index and self.halfIndexBitsMask
  for i in 0 ..< self.roundCount:
    let newLeft = right
    let newRight = left xor self.roundFunction(right)
    left = newLeft
    right = newRight
  result = (left shl self.halfIndexBits) or right

proc decrypt(self: StatelessShuffle, index: uint32): uint32 =
  var left = index shr self.halfIndexBits
  var right = index and self.halfIndexBitsMask
  for i in 0 ..< self.roundCount:
    let newRight = left
    let newLeft = right xor self.roundFunction(left)
    left = newLeft
    right = newRight
  result = (left shl self.halfIndexBits) or right

proc indexToShuffledIndex*(s: StatelessShuffle, index: uint32): uint32 {.inline.} =
  s.encrypt(index)

proc shuffledIndexToIndex*(s: StatelessShuffle, index: uint32): uint32 {.inline.} =
  s.decrypt(index)

import std/random

proc shuffleTest() =
  randomize(0) # Using a deterministic seed for reproducibility
  echo "Shuffling 16 items with 4 rounds"
  var shuffleIterator = StatelessShuffle()
  shuffleIterator.setIndexBits(4)
  shuffleIterator.setRoundCount(4)
  for testIndex in 0 ..< 4:
    let seed = rand(uint32)
    echo "  seed = ", seed
    shuffleIterator.setSeed(seed)
    var first = true
    for index in 0 ..< 16:
      let shuffledIndex = shuffleIterator.indexToShuffledIndex(uint32(index))
      let unshuffledIndex = shuffleIterator.shuffledIndexToIndex(shuffledIndex)
      stdout.write (if first: "" else: ", "), shuffledIndex
      if index != int(unshuffledIndex):
        quit "Error! Round trip failure in shuffleTest"
      first = false
    echo()

  echo "Shuffling 12 items with 4 rounds"
  shuffleIterator = StatelessShuffle()
  shuffleIterator.setIndexBits(4)
  shuffleIterator.setRoundCount(4)
  for testIndex in 0 ..< 4:
    let seed = rand(uint32)
    echo "  seed = ", seed
    shuffleIterator.setSeed(seed)
    var first = true
    for index in 0 ..< 16:
      let shuffledIndex = shuffleIterator.indexToShuffledIndex(uint32(index))

      if shuffledIndex >= 12: continue

      let unshuffledIndex = shuffleIterator.shuffledIndexToIndex(shuffledIndex)
      stdout.write (if first: "" else: ", "), shuffledIndex
      if index != int(unshuffledIndex):
        quit "Error! Round trip failure in shuffleTest"
      first = false
    echo()

# Call the test function
shuffleTest()
