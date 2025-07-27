import std/hashes, bitvec

type
  BloomFilter* = object
    data: BitVector
    salt: seq[int]

proc initBloomFilter*(capacity, hashFuncs: Positive): BloomFilter =
  ## Creates a new BloomFilter with the specified capacity and number of hash functions
  result = BloomFilter(
    data: initBitVector(capacity),
    salt: newSeq[int](hashFuncs)
  )
  # Generate random seeds for hash functions
  for i in 0..<hashFuncs:
    result.salt[i] = i
    # result.salt[i] = rand(int)

proc len*(x: BloomFilter): int {.inline.} =
  ## Returns the capacity of the bloom filter
  x.data.len

proc hashPosition(elemHash: Hash, seed, capacity: int): int {.inline.} =
  ## Compute a single hash position - extracted for inlining
  var h = elemHash
  h = h !& seed
  h = !$h
  result = abs(h) mod capacity

iterator hashPositions(x: BloomFilter, elem: string): int =
  ## Generator that yields index positions returned by the k hash functions
  let elemHash = hash(elem)
  for i in 0..<x.salt.len:
    yield hashPosition(elemHash, x.salt[i], x.len)

proc add*(x: var BloomFilter, elem: string) =
  ## Adds an element to the bloom filter by setting the appropriate bits
  for pos in x.hashPositions(elem):
    x.data.set(pos)

func contains*(x: BloomFilter, elem: string): bool =
  ## Returns true if the element may be contained in the filter, false if definitely not
  result = true
  for pos in x.hashPositions(elem):
    if not x.data.get(pos):
      return false

when isMainModule:
  var x = initBloomFilter(5, 2)

  x.add("hello")
  x.add("bye")

  echo "hello" in x  # Output: true
  echo "bye" in x    # Output: true
  echo "geia" in x   # Output: false, sometimes true (false positive in x
