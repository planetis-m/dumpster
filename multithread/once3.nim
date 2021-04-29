# https://github.com/nim-lang/Nim/pull/16192
# https://www.youtube.com/watch?v=lVBvHbJsg5Y
# https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/double-checked-locking/
# https://preshing.com/20130930/double-checked-locking-is-fixed-in-cpp11/
import std / locks

type
  Once* = object
    L: Lock
    finished: bool

proc initOnce*(o: var Once) =
  initLock(o.L)
  o.finished = false

proc destroyOnce*(o: var Once) {.inline.} =
  deinitLock(o.L)

template once*(o: Once, body: untyped) =
  if not atomicLoadN(addr o.finished, AtomicAcquire):
    acquire o.L
    if not o.finished:
      body
      atomicStoreN(addr o.finished, true, AtomicRelease)
    release o.L

var o: Once
proc smokeOnce() =
  initOnce(o)
  var a = 0
  o.once(a += 1)
  echo a
  assert a == 1
  o.once(a += 1)
  echo a
  assert a == 1
  destroyOnce(o)

smokeOnce()
