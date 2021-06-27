import std/[atomics, isolation]

template Spaces: untyped = Cap + 1

type
  RingBuffer*[Cap: static[int], T] = object
    head, tail: Atomic[int]
    data: array[Spaces, T]

proc push*[Cap, T](this: var RingBuffer[Cap, T]; value: sink Isolated[
    T]): bool {.nodestroy.} =
  let head = this.head.load(moRelaxed)
  var nextHead = head + 1
  if nextHead == Spaces:
    nextHead = 0
  if nextHead == this.tail.load(moAcquire):
    result = false
  else:
    this.data[head] = extract value
    this.head.store(nextHead, moRelease)
    result = true

template push*[Cap, T](this: RingBuffer[Cap, T]; value: T): bool =
  push(this, isolate(value))

proc pop*[Cap, T](this: var RingBuffer[Cap, T]; value: var T): bool =
  let tail = this.tail.load(moRelaxed)
  if tail == this.head.load(moAcquire):
    result = false
  else:
    value = move this.data[tail]
    var nextTail = tail + 1
    if nextTail == Spaces:
      nextTail = 0
    this.tail.store(nextTail, moRelease)
    result = true

when isMainModule:
  proc testBasic =
    var r: RingBuffer[100, int]
    for i in 0..<r.Cap:
      # try to insert an element
      assert r.push(i)
    for i in 0..<r.Cap:
      # try to retrieve an element
      var value: int
      assert r.pop(value)

  testBasic()
