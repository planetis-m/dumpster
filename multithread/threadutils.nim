import locks
{.push stackTrace:off.}

type
  Semaphore* = object
    c: Cond
    L: Lock
    counter: int

proc initSemaphore*(cv: var Semaphore) =
  initCond(cv.c)
  initLock(cv.L)

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

const CacheLineSize = 32 # true for most archs

type
  Barrier* {.compilerproc.} = object
    entered*: int
    cv: Semaphore # Semaphore takes 3 words at least
    when sizeof(int) < 8:
      cacheAlign: array[CacheLineSize-4*sizeof(int), byte]
    left*: int
    cacheAlign2: array[CacheLineSize-sizeof(int), byte]
    interest: bool # whether the master is interested in the "all done" event

proc barrierEnter*(b: var Barrier) {.compilerproc, inline.} =
  # due to the signaling between threads, it is ensured we are the only
  # one with access to 'entered' so we don't need 'atomicInc' here:
  inc b.entered
  # also we need no 'fence' instructions here as soon 'nimArgsPassingDone'
  # will be called which already will perform a fence for us.

proc barrierLeave*(b: var Barrier) {.compilerproc, inline.} =
  atomicInc b.left
  when not defined(x86): fence()
  # We may not have seen the final value of b.entered yet,
  # so we need to check for >= instead of ==.
  if b.interest and b.left >= b.entered: signal(b.cv)

proc openBarrier*(b: var Barrier) {.compilerproc, inline.} =
  b.entered = 0
  b.left = 0
  b.interest = false

proc closeBarrier*(b: var Barrier) {.compilerproc.} =
  fence()
  if b.left != b.entered:
    b.cv.initSemaphore()
    fence()
    b.interest = true
    fence()
    while b.left != b.entered: blockUntil(b.cv)
    destroySemaphore(b.cv)

{.pop.}
