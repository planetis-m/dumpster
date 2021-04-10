import nlocks
{.push stackTrace: off.}

type
  Semaphore* = object
    c: Cond
    L: Lock
    counter: int

proc initSemaphore*(cv: var Semaphore; value: Natural = 0) =
  initCond(cv.c)
  initLock(cv.L)
  cv.counter = value

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
  signal(cv.c)
  release(cv.L)

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
    while cycle == b.cycle:
      wait(b.c, b.L)
  release(b.L)

type
  RwMonitor* = object
    readPhase: Cond
    writePhase: Cond
    L: Lock
    counter: int # can be in three states: free = 0, reading > 0, writing = -1

proc initRwMonitor*(rw: var RwMonitor) =
  initCond rw.readPhase
  initCond rw.writePhase
  initLock rw.L
  rw.counter = 0

proc destroyRwMonitor*(rw: var RwMonitor) {.inline.} =
  deinitCond(rw.readPhase)
  deinitCond(rw.writePhase)
  deinitLock(rw.L)

proc beginRead*(rw: var RwMonitor) =
  acquire(rw.L)
  while rw.counter == -1:
    wait(rw.readPhase, rw.L)
  inc rw.counter
  release(rw.L)

proc beginWrite*(rw: var RwMonitor) =
  acquire(rw.L)
  while rw.counter != 0:
    wait(rw.writePhase, rw.L)
  rw.counter = -1
  release(rw.L)

proc endRead*(rw: var RwMonitor) =
  acquire(rw.L)
  dec rw.counter
  if rw.counter == 0:
    rw.writePhase.signal()
  release(rw.L)

proc endWrite*(rw: var RwMonitor) =
  acquire(rw.L)
  rw.counter = 0
  rw.readPhase.broadcast()
  rw.writePhase.signal()
  release(rw.L)

template readWith*(a: RwMonitor, body: untyped) =
  mixin beginRead, endRead
  beginRead(a)
  try:
    body
  finally:
    endRead(a)

template writeWith*(a: RwMonitor, body: untyped) =
  mixin beginWrite, endWrite
  beginWrite(a)
  try:
    body
  finally:
    endWrite(a)

{.pop.}
