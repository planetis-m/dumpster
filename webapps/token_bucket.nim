import std/times, os

type
  TokenBucket = object
    capacity, tokens, refillRate: float
    lastRefill: float

proc refill(tb: var TokenBucket) =
  let now = epochTime()
  let elapsedSeconds = now - tb.lastRefill
  let refillAmount = elapsedSeconds * tb.refillRate
  tb.tokens = min(tb.capacity, tb.tokens + refillAmount)
  tb.lastRefill = now

proc consume(tb: var TokenBucket; tokens: float): bool =
  tb.refill()
  echo tb.tokens
  if tokens <= tb.tokens:
    tb.tokens -= tokens
    true
  else:
    false

proc newTokenBucket(capacity: float; refillRate: float = 0): TokenBucket =
  TokenBucket(capacity: capacity, tokens: capacity, refillRate: refillRate, lastRefill: epochTime())

var
  tokenBucket = newTokenBucket(10, 1)

var count = 0
for i in 1..180:
  sleep(400)
  var res = tokenBucket.consume(1)
  if res: inc count
  echo res
echo count
