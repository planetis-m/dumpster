import std/times

type
  SlidingWindow* = object # Approximate
    capacity: int
    currentCount, previousCount: int
    windowSize: float
    currentTime: float

proc allowRequest*(sw: var SlidingWindow): bool =
  let now = epochTime()
  # If the current time is outside the window, reset the window
  if now - sw.currentTime > sw.windowSize:
    sw.currentTime = now
    sw.previousCount = sw.currentCount
    sw.currentCount = 0
  # Calculate the weighted average of the previous and current counts
  let weight = (sw.windowSize - (now - sw.currentTime)) / sw.windowSize
  let estimatedCount = int(sw.previousCount.float * weight) + sw.currentCount
  # Check if the count exceeds the capacity
  if estimatedCount <= sw.capacity:
    # Increment the current count and allow the request
    inc sw.currentCount
    true
  else:
    false

proc newSlidingWindow*(capacity: Positive, windowSize: Duration): SlidingWindow =
  SlidingWindow(
    capacity: capacity,
    previousCount: capacity, currentCount: 0,
    windowSize: windowSize,
    currentTime: getTime()
  )

when isMainModule:
  import std/os

  var
    slidingWindow = newSlidingWindow(1, 1)

  var count = 0
  for i in 1..180:
    sleep(400)
    var res = slidingWindow.allowRequest()
    if res: inc count
    echo res
  echo count
