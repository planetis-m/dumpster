# http://byronlai.com/jekyll/update/2015/12/26/barrier.html
# https://stackoverflow.com/questions/9815798/how-to-find-barrier-functions-implementation
import nlocks

type
  Barrier* = object
    L: Lock
    c: Cond
    threadsRequired: int # number of threads needed for the barrier to continue
    left: int # current barrier count, # of threads still needed.
    cycle: uint # generation count

proc initBarrier*(b: var Barrier; count: int) =
  b.threadsRequired = count
  b.left = count
  b.cycle = 0
  initLock(b.L)
  initCond(b.c)

proc destroyBarrier*(b: var Barrier) {.inline.} =
  deinitLock(b.L)
  deinitCond(b.c)

proc wait*(b: var Barrier) =
  acquire(b.L)
  dec b.left
  if b.left == 0:
    inc b.cycle
    b.left = b.threadsRequired
    broadcast(b.c)
  else:
    let cycle = b.cycle
    while cycle == b.cycle: wait(b.c, b.L)
  release(b.L)
