# https://stackoverflow.com/questions/9854470/implementing-a-binary-semaphore-class-in-c
import std / locks

type
  Event* = object
    c: Cond
    L: Lock
    signaled: bool

proc `=destroy`*(s: var Event) =
  deinitCond(s.c)
  deinitLock(s.L)

proc `=sink`*(dest: var Event; source: Event) {.error.}
proc `=copy`*(dest: var Event; source: Event) {.error.}

proc initEvent*(s: var Event; value = false) =
  s.signaled = value
  initCond(s.c)
  initLock(s.L)

proc wait*(s: var Event) =
  acquire(s.L)
  while not s.signaled:
    wait(s.c, s.L)
  s.signaled = false
  release(s.L)

proc signal*(s: var Event) =
  acquire(s.L)
  s.signaled = true
  signal(s.c)
  release(s.L)
