# https://www.ea.com/seed/news/constant-time-stateless-shuffling
type
  StatelessShuffle = object
    roundCount: uint32 # The number of rounds in a Feistel network is tuneable for quality versus speed.
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

proc toShuffledIdx*(s: StatelessShuffle, index: uint32): uint32 {.inline.} =
  s.encrypt(index)

proc fromShuffledIdx*(s: StatelessShuffle, index: uint32): uint32 {.inline.} =
  s.decrypt(index)

import std/math

proc shuffle*[T](s: StatelessShuffle, x: var openarray[T]) =
  for i in 0..<nextPowerOfTwo(x.len):
    let j = s.toShuffledIdx(i.uint32)
    if j.int < x.len:
      assert i.uint32 == s.fromShuffledIdx(j), "roundtrip failure"
      swap(x[i], x[j])

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
      let shuffledIndex = shuffleIterator.toShuffledIdx(index.uint32)
      let unshuffledIndex = shuffleIterator.fromShuffledIdx(shuffledIndex)
      stdout.write (if first: "" else: ", "), shuffledIndex
      if index.uint32 != unshuffledIndex:
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
      let shuffledIndex = shuffleIterator.toShuffledIdx(index.uint32)
      if shuffledIndex >= 12: continue
      let unshuffledIndex = shuffleIterator.fromShuffledIdx(shuffledIndex)
      stdout.write (if first: "" else: ", "), shuffledIndex
      if index.uint32 != unshuffledIndex:
        quit "Error! Round trip failure in shuffleTest"
      first = false
    echo()

shuffleTest()

proc getRequiredBits(length: uint32): uint32 {.inline.} =
  result = ceil(log2(float(length))).uint32
  if (result and 1) != 0:
    inc(result)

const times = 100_000

randomize()
proc frequencyTest[Size: static int]() =
  var frequencies: array[Size, array[Size, int]] # Position frequencies

  var s = StatelessShuffle()
  let requiredBits = getRequiredBits(Size)
  s.setIndexBits(requiredBits)
  s.setRoundCount(4)

  for _ in 0..<times:
    let seed = rand(uint32)
    s.setSeed(seed)
    var arr: array[Size, int]
    for i in 0..<Size:
      arr[i] = i
    s.shuffle(arr)
    for i in 0..<Size:
      frequencies[i][arr[i]].inc

  let expectedFrequency = times div Size
  let tolerance = expectedFrequency.float / 4 # 25% tolerance

  for i in 0..<Size:
    for j in 0..<Size:
      doAssert abs(frequencies[i][j] - expectedFrequency).float <= tolerance, "Frequency test failed"

frequencyTest[16]()
# frequencyTest[128]()
# frequencyTest[256]()
# frequencyTest[260]()
