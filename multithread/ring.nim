import std/[math, isolation]

type
  Ringbuffer*[Cap: static[int], T] = object
    ring: array[Cap, T]
    head, tail: int # atomic

template next(current: untyped): untyped = (current + 1) and Cap - 1

proc push*[Cap, T](this: var Ringbuffer[Cap, T]; value: sink Isolated[T]): bool =
  assert isPowerOfTwo(Cap)
  let head = atomicLoadN(addr this.head, AtomicRelaxed)
  let nextHead = next(head)
  if nextHead == atomicLoadN(addr this.tail, AtomicAcquire):
    result = false
  else:
    this.ring[head] = extract value
    atomicStoreN(addr this.head, nextHead, AtomicRelease)
    result = true

template push*[Cap, T](this: Ringbuffer[Cap, T]; value: T): bool =
  push(this, isolate(value))

proc pop*[Cap, T](this: var Ringbuffer[Cap, T]; value: var T): bool =
  assert isPowerOfTwo(Cap)
  let tail = atomicLoadN(addr this.tail, AtomicRelaxed)
  if tail == atomicLoadN(addr this.head, AtomicAcquire):
    result = false
  else:
    value = move this.ring[tail]
    atomicStoreN(addr this.tail, next(tail), AtomicRelease)
    result = true

const
  seed = 99
  bufCap = 16
  numIters = 1000

var
  rng: Ringbuffer[bufCap, int]
  thr1, thr2: Thread[void]

proc producer =
  for i in 0 ..< numIters:
    while not rng.push(i + seed): cpuRelax()
    #echo " >> pushed ", i+seed

proc consumer =
  for i in 0 ..< numIters:
    var res: int
    while not rng.pop(res): cpuRelax()
    #echo " >> popped ", res
    assert res == seed + i

proc testSpScRing =
  createThread(thr1, producer)
  createThread(thr2, consumer)
  joinThread(thr1)
  joinThread(thr2)

testSpScRing()
