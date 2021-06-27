import std/[atomics, isolation]
from std/typetraits import supportsCopyMem

const
  cacheLineSize = 64

type
  SpscQueue*[T] = object
    cap: int
    data: ptr UncheckedArray[T]
    head, tail {.align(cacheLineSize).}: Atomic[int]

proc `=destroy`*[T](this: var SpscQueue[T]) =
  if this.data != nil:
    when not supportsCopyMem(T):
      var tail = this.tail.load(moRelaxed)
      let head = this.head.load(moAcquire)
      while tail != head:
        `=destroy`(this.data[tail])
        inc tail
        if tail == this.cap:
          tail = 0
    deallocShared(this.data)

proc `=sink`*[T](dest: var SpscQueue[T]; source: SpscQueue[T]) {.error.}
proc `=copy`*[T](dest: var SpscQueue[T]; source: SpscQueue[T]) {.error.}

proc init*[T](this: var SpscQueue[T]; capacity: Natural) =
  this.cap = capacity + 1
  this.data = cast[ptr UncheckedArray[T]](allocShared(this.cap))

proc cap*[T](this: SpscQueue[T]): int = this.cap - 1

proc len*[T](this: SpscQueue[T]): int =
  result = this.head.load(moAcquire) - this.tail.load(moAcquire)
  if result < 0:
    result += this.cap

proc push*[T](this: var SpscQueue[T]; value: sink Isolated[T]): bool {.
    nodestroy.} =
  let head = this.head.load(moRelaxed)
  var nextHead = head + 1
  if nextHead == this.cap:
    nextHead = 0
  if nextHead == this.tail.load(moAcquire):
    result = false
  else:
    this.data[head] = extract value
    this.head.store(nextHead, moRelease)
    result = true

template push*[T](this: SpscQueue[T]; value: T): bool =
  push(this, isolate(value))

proc pop*[T](this: var SpscQueue[T]; value: var T): bool =
  let tail = this.tail.load(moRelaxed)
  if tail == this.head.load(moAcquire):
    result = false
  else:
    value = move this.data[tail]
    var nextTail = tail + 1
    if nextTail == this.cap:
      nextTail = 0
    this.tail.store(nextTail, moRelease)
    result = true

when isMainModule:
  proc testBasic =
    var r: SpscQueue[int]
    init(r, 100)
    for i in 0..<r.cap:
      # try to insert an element
      if r.push(i):
        # succeeded
        discard
      else:
        # buffer full
        assert i == r.cap - 1
    for i in 0..<r.cap:
      # try to retrieve an element
      var value: int
      if r.pop(value):
        # succeeded
        discard
      else:
        # buffer empty
        assert i == r.cap - 1

  testBasic()
