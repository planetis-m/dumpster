import std/locks

type
  Event* = object
    c1: Cond
    c2: Cond
    L: Lock
    counter: int
    cycle: uint
    signaled: bool

proc `=destroy`*(s: var Event) =
  deinitCond(s.c1)
  deinitCond(s.c2)
  deinitLock(s.L)

proc `=sink`*(dest: var Event; source: Event) {.error.}
proc `=copy`*(dest: var Event; source: Event) {.error.}

proc init*(s: var Event; value = false) =
  s.signaled = value
  s.counter = 0
  s.cycle = 0
  initCond(s.c1)
  initCond(s.c2)
  initLock(s.L)

proc wait*(s: var Event) =
  acquire(s.L)
  if not s.signaled:
    inc s.counter
    let cycle = s.cycle
    while true:
      wait(s.c1, s.L)
      if s.signaled or cycle != s.cycle: break
    dec s.counter
    if s.counter == 0:
      signal(s.c2)
  release(s.L)

proc signal*(s: var Event) =
  acquire(s.L)
  s.signaled = true
  inc s.cycle
  broadcast(s.c1)
  while s.counter > 0:
    wait(s.c2, s.L)
  s.signaled = false
  release(s.L)
