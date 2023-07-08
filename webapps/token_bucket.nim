import std/[times, tables]

type
  TokenBucket*[K] = object
    capacity, refillRate: float
    buckets: Table[K, tuple[tokens, lastRefillTime: float]]

proc refill(bucket: var tuple[tokens, lastRefillTime: float], capacity, refillRate: float) =
  let now = epochTime()
  let elapsedSeconds = now - bucket.lastRefillTime
  let refillAmount = elapsedSeconds * refillRate
  bucket = (min(capacity, bucket.tokens + refillAmount), now)

proc refill[K](tb: var TokenBucket[K]; key: K) =
  if tb.buckets.hasKeyOrPut(key, (tb.capacity, epochTime())):
    tb.buckets[key].refill(tb.capacity, tb.refillRate)

proc consume(tokensInBucket: var float, tokens: float): bool =
  if tokens <= tokensInBucket:
    tokensInBucket -= tokens
    true
  else:
    false

proc consume*[K](tb: var TokenBucket[K]; key: K; tokens: float): bool =
  tb.refill(key)
  tb.buckets[key].tokens.consume(tokens)

proc newTokenBucket*[K](capacity, refillRate: float): TokenBucket[K] =
  TokenBucket[K](capacity: capacity, refillRate: refillRate)

import std/os

var
  tokenBucket = newTokenBucket[string](10, 1)

var count = 0
for i in 1..180:
  sleep(400)
  var res = tokenBucket.consume("a", 1)
  if res: inc count
  echo res
echo count
