import std/math

type
  RingBuffer*[Cap: static[int], T] = object
    head, tail: int # atomic
    data: array[Cap, T]

template next(current: untyped): untyped = (current + 1) and Cap - 1

proc push*[Cap, T](this: var RingBuffer[Cap, T]; value: sink T): bool =
  assert isPowerOfTwo(Cap)
  let head = atomicLoadN(addr this.head, AtomicRelaxed)
  let nextHead = next(head)
  if nextHead == atomicLoadN(addr this.tail, AtomicAcquire):
    result = false
  else:
    this.data[head] = value
    atomicStoreN(addr this.head, nextHead, AtomicRelease)
    result = true

proc pop*[Cap, T](this: var RingBuffer[Cap, T]; value: var T): bool =
  assert isPowerOfTwo(Cap)
  let tail = atomicLoadN(addr this.tail, AtomicRelaxed)
  if tail == atomicLoadN(addr this.head, AtomicAcquire):
    result = false
  else:
    value = move this.data[tail]
    atomicStoreN(addr this.tail, next(tail), AtomicRelease)
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
        discard

  testBasic()
