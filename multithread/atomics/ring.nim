import std/[atomics, math, isolation]

type
  RingBuffer*[Cap: static[int], T] = object
    head, tail: Atomic[int]
    data: array[Cap, T]

template next(current: untyped): untyped = (current + 1) and Cap - 1

proc push*[Cap, T](this: var RingBuffer[Cap, T]; value: sink Isolated[
    T]): bool {.nodestroy.} =
  assert isPowerOfTwo(Cap)
  let head = this.head.load(moRelaxed)
  let nextHead = next(head)
  if nextHead == this.tail.load(moAcquire):
    result = false
  else:
    this.data[head] = extract value
    this.head.store(nextHead, moRelease)
    result = true

template push*[Cap, T](this: RingBuffer[Cap, T]; value: T): bool =
  push(this, isolate(value))

proc pop*[Cap, T](this: var RingBuffer[Cap, T]; value: var T): bool =
  assert isPowerOfTwo(Cap)
  let tail = this.tail.load(moRelaxed)
  if tail == this.head.load(moAcquire):
    result = false
  else:
    value = move this.data[tail]
    this.tail.store(next(tail), moRelease)
    result = true

when isMainModule:
  proc testBasic =
    var r: RingBuffer[32, int]
    for i in 0..<r.Cap:
      # try to insert an element
      if r.push(i):
        # succeeded
        discard
      else:
        assert i == r.Cap - 1
        # buffer full
    for i in 0..<r.Cap:
      # try to retrieve an element
      var value: int
      if r.pop(value):
        # succeeded
        discard
      else:
        # buffer empty
        assert i == r.Cap - 1

  testBasic()
