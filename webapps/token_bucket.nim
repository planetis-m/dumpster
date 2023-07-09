import std/[times, tables]

# type
#   TokenBucket*[K] = object
#     capacity, refillRate: float
#     buckets: Table[K, tuple[tokens, lastRefillTime: float]]
#
# proc refill[K](tb: var TokenBucket[K]; key: K) =
#   if tb.buckets.hasKeyOrPut(key, (tb.capacity, epochTime())):
#     var bucket = addr tb.buckets[key]
#     let now = epochTime()
#     let elapsedSeconds = now - bucket.lastRefillTime
#     let refillAmount = elapsedSeconds * tb.refillRate
#     bucket[] = (min(tb.capacity, bucket.tokens + refillAmount), now)
#
# proc consume*[K](tb: var TokenBucket[K]; key: K; tokens: float): bool =
#   tb.refill(key)
#   var tokensInBucket = addr tb.buckets[key].tokens
#   if tokens <= tokensInBucket[]:
#     tokensInBucket[] -= tokens
#     true
#   else:
#     false
#
# proc newTokenBucket*[K](capacity, refillRate: float): TokenBucket[K] =
#   TokenBucket[K](capacity: capacity, refillRate: refillRate)

type
  TokenBucket = object
    capacity, tokens, refillRate: float
    lastRefillTime: float

proc refill(tb: var TokenBucket) =
  let now = epochTime()
  let elapsedSeconds = now - tb.lastRefillTime
  let refillAmount = elapsedSeconds * tb.refillRate
  tb.tokens = min(tb.capacity, tb.tokens + refillAmount)
  tb.lastRefillTime = now

proc consume(tb: var TokenBucket; tokens: float): bool =
  # if tb.tokens >= tokens:
  #   tb.tokens -= tokens
  #   return true
  tb.refill()
  if tokens <= tb.tokens:
    tb.tokens -= tokens
    true
  else:
    false

proc newTokenBucket(capacity, refillRate: float): TokenBucket =
  TokenBucket(capacity: capacity, tokens: capacity, refillRate: refillRate, lastRefillTime: epochTime())

import std/os

var
  tokenBucket = newTokenBucket(10, 1)

var count = 0
for i in 1..180:
  sleep(400)
  var res = tokenBucket.consume(1)
  if res: inc count
  echo res
echo count
