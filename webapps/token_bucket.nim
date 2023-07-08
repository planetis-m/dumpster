import std/times, os

type
  TokenBucket = object
    capacity, tokens, refillRate: float
    lastRefill: Time

proc refill(tb: var TokenBucket) =
  let now = getTime()
  let elapsedSeconds = now - tb.lastRefill
  let refillAmount = elapsedSeconds.inSeconds.float * tb.refillRate # This is problematic...
  tb.tokens = min(tb.capacity, tb.tokens + refillAmount)
  tb.lastRefill = now

proc consume(tb: var TokenBucket; tokens: float): bool =
  tb.refill()
  if tokens <= tb.tokens:
    tb.tokens -= tokens
    true
  else:
    false

proc newTokenBucket(capacity, refillRate: float): TokenBucket =
  TokenBucket(capacity: capacity, tokens: capacity, refillRate: refillRate, lastRefill: getTime())

var
  tokenBucket = newTokenBucket(10, 1)

var count = 0
for i in 1..180:
  sleep(400)
  var res = tokenBucket.consume(1)
  if res: inc count
  echo res
echo count
