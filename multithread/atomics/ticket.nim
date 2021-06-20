# https://mfukar.github.io/2017/09/08/ticketspinlock.html
# https://www.youtube.com/watch?v=bZv3wctQaUA
# https://www.youtube.com/watch?v=r8eNGLY26T0
const
  CacheLineSize = 64
  BackoffMin = 2

type
  TicketLock* = object
    serving {.align(CacheLineSize).}: uint
    line {.align(CacheLineSize).}: uint

proc acquire*(t: var TicketLock) =
  let place = atomicFetchAdd(addr t.line, 1, AtomicRelaxed)
  while atomicLoadN(addr t.serving, AtomicAcquire) != place: cpuRelax()

proc acquireBackoff*(t: var TicketLock) =
  let place = atomicFetchAdd(addr t.line, 1, AtomicRelaxed)
  while true:
    let serving = atomicLoadN(addr t.serving, AtomicAcquire)
    if serving == place: return
    let previous = place - serving
    let delay = BackoffMin * previous
    for i in 0 ..< delay: cpuRelax()

proc release*(t: var TicketLock) =
  let next = atomicLoadN(addr t.serving, AtomicRelaxed) + 1
  atomicStoreN(addr t.serving, next, AtomicRelease)

template withLock*(a: TicketLock, body: untyped) =
  acquireBackoff(a)
  try:
    body
  finally:
    release(a)
