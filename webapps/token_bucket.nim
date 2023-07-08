import std/[times, tables]

type
  TokenBucket[K] = object
    capacity, refillRate: float
    buckets: Table[K, tuple[tokens: float, lastRefillTime: float]]

proc refill[K](tb: var TokenBucket[K]; key: K) =
  if tb.buckets.hasKeyOrPut(key, (tb.capacity, epochTime())):
    let (tokens, lastRefillTime) = tb.buckets[key]
    let now = epochTime()
    let elapsedSeconds = now - lastRefillTime
    let refillAmount = elapsedSeconds * tb.refillRate
    tb.buckets[key] = (min(tb.capacity, tokens + refillAmount), now)

proc consume[K](tb: var TokenBucket[K]; key: K; tokens: float): bool =
  tb.refill(key)
  if tokens <= tb.buckets[key].tokens:
    tb.buckets[key].tokens -= tokens
    true
  else:
    false

proc newTokenBucket[K](capacity, refillRate: float): TokenBucket[K] =
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
