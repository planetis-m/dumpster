# https://stackoverflow.com/questions/9854470/implementing-a-binary-semaphore-class-in-c
import std / locks

type
  BinSemaphore* = object
    c: Cond
    L: Lock
    signaled: bool

proc initBinSemaphore*(s: var BinSemaphore; value = false) =
  initCond(s.c)
  initLock(s.L)
  s.signaled = value

proc destroyBinSemaphore*(s: var BinSemaphore) {.inline.} =
  deinitCond(s.c)
  deinitLock(s.L)

proc blockUntil*(s: var BinSemaphore) =
  acquire(s.L)
  while not s.signaled:
    wait(s.c, s.L)
  s.signaled = false
  release(s.L)

proc signal*(s: var BinSemaphore) =
  acquire(s.L)
  let prev = s.signaled
  s.signaled = true
  if not prev:
    signal(s.c)
  release(s.L)
