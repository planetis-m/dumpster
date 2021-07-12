# https://mfukar.github.io/2017/09/08/ticketspinlock.html
# https://www.youtube.com/watch?v=bZv3wctQaUA
# https://www.youtube.com/watch?v=r8eNGLY26T0
import sync/atomics2

const
  CacheLineSize = 64
  BackoffMin = 2

type
  TicketLock* = object
    serving {.align(CacheLineSize).}: Atomic[uint]
    line {.align(CacheLineSize).}: Atomic[uint]

proc acquire*(t: var TicketLock) =
  let place = fetchAdd(t.line, 1, Relaxed)
  while t.serving.load(Acquire) != place: cpuRelax()

proc acquireBackoff*(t: var TicketLock) =
  let place = fetchAdd(t.line, 1, Relaxed)
  while true:
    let serving = t.serving.load(Acquire)
    if serving == place: return
    let previous = place - serving
    let delay = BackoffMin * previous
    for i in 0 ..< delay: cpuRelax()

proc release*(t: var TicketLock) =
  let next = t.serving.load(Relaxed) + 1
  t.serving.store(next, Release)

template withLock*(a: TicketLock, body: untyped) =
  acquireBackoff(a)
  {.locks: [a].}:
    try:
      body
    finally:
      release(a)
