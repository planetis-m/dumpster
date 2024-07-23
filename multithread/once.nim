# https://github.com/llvm/llvm-project/tree/main/libc/src/threads

type
  Once* = distinct int

const
  Incomplete = 0
  Running = 1
  Complete = 2

template once*(o: Once, body: untyped) =
  var expected = Incomplete
  if atomicLoadN(addr o.int, AtomicRelaxed) == Incomplete and
      atomicCompareExchangeN(addr o.int, addr expected, Running, false,
          AtomicAcquire, AtomicRelaxed):
    body
    atomicStoreN(addr o.int, Complete, AtomicRelease)
  else:
    while atomicLoadN(addr o.int, AtomicAcquire) != Complete: cpuRelax()

var o: Once
proc smokeOnce() =
  var a = 0
  o.once(a += 1)
  #echo a
  assert a == 1
  o.once(a += 1)
  #echo a
  assert a == 1

smokeOnce()
