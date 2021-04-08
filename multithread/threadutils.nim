import nlocks

{.push stackTrace: off.}

type
  Semaphore* = object
    c: Cond
    L: Lock
    counter: int

proc initSemaphore*(cv: var Semaphore; counter = 0) =
  initCond(cv.c)
  initLock(cv.L)
  cv.counter = counter

proc destroySemaphore*(cv: var Semaphore) {.inline.} =
  deinitCond(cv.c)
  deinitLock(cv.L)

proc blockUntil*(cv: var Semaphore) =
  acquire(cv.L)
  while cv.counter <= 0:
    wait(cv.c, cv.L)
  dec cv.counter
  release(cv.L)

proc signal*(cv: var Semaphore) =
  acquire(cv.L)
  inc cv.counter
  release(cv.L)
  signal(cv.c)

type
  Barrier* = object
    c: Cond
    L: Lock
    counter: int
    maxThreads: int
    globalSense: bool

proc initBarrier*(b: var Barrier; numThreads = high(int)) =
  # When `numThreads` isn't specified, it has to be passed when syncing `sync(numThreads)`.
  initCond(b.c)
  initLock(b.L)
  b.counter = numThreads
  b.maxThreads = numThreads
  b.globalSense = false

proc destroyBarrier*(b: var Barrier) {.inline.} =
  deinitCond(b.c)
  deinitLock(b.L)

proc sync*(b: var Barrier) =
  assert b.maxThreads < high(int)
  let localSense = not b.globalSense
  acquire(b.L)
  dec b.counter
  if b.counter == 0:
    b.counter = b.maxThreads
    b.globalSense = localSense
    broadcast(b.c)
  else:
    #while b.globalSense != localSense:
    wait(b.c, b.L)
    assert b.globalSense == localSense
  release(b.L)

proc sync*(b: var Barrier; numThreads: int) =
  assert numThreads <= b.maxThreads
  let localSense = not b.globalSense
  acquire(b.L)
  dec b.counter
  if b.counter == b.maxThreads - numThreads:
    b.counter = b.maxThreads
    b.globalSense = localSense
    broadcast(b.c)
  else:
    #while b.globalSense != localSense:
    wait(b.c, b.L)
    assert b.globalSense == localSense
  release(b.L)

{.pop.}
