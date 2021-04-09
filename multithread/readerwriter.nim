# http://www.cs.cornell.edu/courses/cs4410/2017su/lectures/lec08-rw.html
import nlocks

type
  RwMonitor* = object
    noWriters: Cond
    noRw: Cond
    L: Lock
    numReaders: int
    isWriter: bool

proc initRwMonitor*(rw: var RwMonitor) =
  initLock rw.L
  rw.numReaders = 0
  rw.isWriter = false
  # predicate: not rw.isWriter
  initCond rw.noWriters
  # predicate: rw.numReaders == 0 and not rw.isWriter
  # NOTE: safe to call signal, waiter must invalidate
  initCond rw.noRw

proc destroyRwMonitor*(rw: var RwMonitor) {.inline.} =
  deinitCond(rw.noWriters)
  deinitCond(rw.noRw)
  deinitLock(rw.L)

proc beginRead*(rw: var RwMonitor) =
  acquire(rw.L)
  while rw.isWriter:
    wait(rw.noWriters, rw.L)
  inc rw.numReaders
  # no need to signal
  release(rw.L)

proc beginWrite*(rw: var RwMonitor) =
  acquire(rw.L)
  while rw.numReaders > 0 or rw.isWriter:
    wait(rw.noRw, rw.L)
  rw.isWriter = true
  # no need to signal
  release(rw.L)

proc endRead*(rw: var RwMonitor) =
  acquire(rw.L)
  dec rw.numReaders
  if rw.numReaders == 0:
    rw.noRw.signal()
  release(rw.L)

proc endWrite*(rw: var RwMonitor) =
  acquire(rw.L)
  rw.isWriter = false
  # note: noRw must not have been true before because of precondition
  # note: noRw guaranteed because invariant says readers == 0
  # note: signal is safe because noRw guaranteed to be invalidated
  rw.noRw.signal()
  rw.noWriters.broadcast()
  release(rw.L)

template reader*(a: RwMonitor, body: untyped) =
  mixin beginRead, endRead
  beginRead(a)
  try:
    body
  finally:
    endRead(a)

template writer*(a: RwMonitor, body: untyped) =
  mixin beginWrite, endWrite
  beginWrite(a)
  try:
    body
  finally:
    endWrite(a)
