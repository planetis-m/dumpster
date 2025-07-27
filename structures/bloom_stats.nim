import std/math, blfilter

proc estimateItems*(bf: BloomFilter): float =
  ## Estimates the number of items in the bloom filter
  ## Formula: n ≈ -(m/k) * ln(1 - X/m)
  ## where m = capacity, k = hash functions, X = number of set bits
  let setBits = bf.data.popcount()
  if setBits == 0:
    return 0.0
  if setBits == bf.len:
    return float.high  # Saturated

  let m = float(bf.len)
  let k = float(bf.salt.len)
  let x = float(setBits)

  result = -(m / k) * ln(1.0 - x / m)

proc falsePositiveRate*(bf: BloomFilter, estimatedItems: float): float =
  ## Estimates the current false positive rate given an estimated item count
  ## Formula: (1 - e^(-kn/m))^k
  if estimatedItems <= 0:
    return 0.0

  let m = float(bf.len)
  let k = float(bf.salt.len)
  let n = estimatedItems

  result = pow(1.0 - exp(-k * n / m), k)

proc falsePositiveRate*(bf: BloomFilter): float =
  ## Estimates the current false positive rate using estimated item count
  bf.falsePositiveRate(bf.estimateItems())

proc optimalCapacity*(expectedItems: int, falsePositiveRate: float): int =
  ## Calculates optimal bit array size for given parameters
  ## Formula: m = -n * ln(p) / (ln(2)^2)
  if falsePositiveRate <= 0.0 or falsePositiveRate >= 1.0:
    raise newException(ValueError, "False positive rate must be between 0 and 1")

  result = int(-float(expectedItems) * ln(falsePositiveRate) / (ln(2.0) * ln(2.0)))

proc optimalHashFunctions*(expectedItems, capacity: int): int =
  ## Calculates optimal number of hash functions for given parameters
  ## Formula: k = (m/n) * ln(2)
  result = max(1, int(float(capacity) / float(expectedItems) * ln(2.0)))

proc newOptimalBloomFilter*(expectedItems: int, falsePositiveRate: float): BloomFilter =
  ## Creates a bloom filter optimally sized for the expected number of items
  ## and desired false positive rate
  let capacity = optimalCapacity(expectedItems, falsePositiveRate)
  let hashFuncs = optimalHashFunctions(expectedItems, capacity)

  result = initBloomFilter(capacity, hashFuncs)
