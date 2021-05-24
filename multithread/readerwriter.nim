# http://www.cs.cornell.edu/courses/cs4410/2017su/lectures/lec08-rw.html
import std/[locks, typetraits]

type
  RwMonitor*[T] = object
    noWriters: Cond
    noRw: Cond
    L: Lock
    numReaders: int
    isWriter: bool
    val: T

proc initRwMonitor*[T](rw: var RwMonitor[T]; val: sink T) {.nodestroy.} =
  initLock rw.L
  rw.numReaders = 0
  rw.isWriter = false
  # predicate: not rw.isWriter
  initCond rw.noWriters
  # predicate: rw.numReaders == 0 and not rw.isWriter
  # NOTE: safe to call signal, waiter must invalidate
  initCond rw.noRw
  rw.val = val

proc `=destroy`*[T](rw: var RwMonitor[T]) {.inline.} =
  deinitCond(rw.noWriters)
  deinitCond(rw.noRw)
  deinitLock(rw.L)
  `=destroy`(rw.val)

proc beginRead*[T](rw: var RwMonitor[T]) =
  acquire(rw.L)
  while rw.isWriter:
    wait(rw.noWriters, rw.L)
  inc rw.numReaders
  # no need to signal
  release(rw.L)

proc beginWrite*[T](rw: var RwMonitor[T]) =
  acquire(rw.L)
  while rw.numReaders > 0 or rw.isWriter:
    wait(rw.noRw, rw.L)
  rw.isWriter = true
  # no need to signal
  release(rw.L)

proc endRead*[T](rw: var RwMonitor[T]) =
  acquire(rw.L)
  dec rw.numReaders
  if rw.numReaders == 0:
    rw.noRw.signal()
  release(rw.L)

proc endWrite*[T](rw: var RwMonitor[T]) =
  acquire(rw.L)
  rw.isWriter = false
  # note: noRw must not have been true before because of precondition
  # note: noRw guaranteed because invariant says readers == 0
  # note: signal is safe because noRw guaranteed to be invalidated
  rw.noRw.signal()
  rw.noWriters.broadcast()
  release(rw.L)

type
  RwReader*[T] = distinct ptr RwMonitor[T]

proc `=destroy`*[T](x: var RwReader[T]) =
  endRead(distinctBase(x)[])

proc read*[T](x: var RwMonitor[T]): RwReader[T] =
  beginRead(x)
  result = RwReader[T](addr x)

proc val*[T](x: RwReader[T]): lent T = distinctBase(x).val

type
  RwWriter*[T] = distinct ptr RwMonitor[T]

proc `=destroy`*[T](x: var RwWriter[T]) =
  endWrite(distinctBase(x)[])

proc write*[T](x: var RwMonitor[T]): RwWriter[T] =
  beginWrite(x)
  result = RwWriter[T](addr x)

proc val*[T](x: RwWriter[T]): var T = distinctBase(x).val
proc `val=`*[T](x: RwWriter[T]; val: sink T) = distinctBase(x).val = val
