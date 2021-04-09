import nlocks
{.push stackTrace: off.}

type
  Semaphore* = object
    c: Cond
    L: Lock
    counter: int

proc initSemaphore*(cv: var Semaphore; counter: Natural = 0) =
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
    required: int # number of threads needed for the barrier to continue
    left: int # current barrier count, number of threads still needed.
    cycle: uint # generation count

proc initBarrier*(b: var Barrier; count: Natural) =
  b.required = count
  b.left = count
  b.cycle = 0
  initCond(b.c)
  initLock(b.L)

proc destroyBarrier*(b: var Barrier) {.inline.} =
  deinitCond(b.c)
  deinitLock(b.L)

proc wait*(b: var Barrier) =
  acquire(b.L)
  dec b.left
  if b.left == 0:
    inc b.cycle
    b.left = b.required
    broadcast(b.c)
  else:
    let cycle = b.cycle
    while cycle == b.cycle: wait(b.c, b.L)
  release(b.L)

{.pop.}
