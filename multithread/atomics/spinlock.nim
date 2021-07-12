# https://rigtorp.se/spinlock/
type
  SpinLock* = object
    lock: bool

proc `=sink`*(dest: var SpinLock; source: SpinLock) {.error.}
proc `=copy`*(dest: var SpinLock; source: SpinLock) {.error.}

proc acquire(s: var SpinLock) =
  while true:
    # Optimistically assume the lock is free on the first try.
    if not atomicExchangeN(addr s.lock, true, AtomicAcquire):
      return
    else:
      # Wait for lock to be released without generating cache misses.
      while atomicLoadN(addr s.lock, AtomicRelaxed):
        # Reduces contention between hyper-threads.
        cpuRelax()

proc tryAcquire(s: var SpinLock): bool =
  # First do a relaxed load to check if lock is free in order to prevent
  # unnecessary cache misses if someone does `while not tryAcquire(s)`
  result = not atomicLoadN(addr s.lock, AtomicRelaxed) and
      not atomicExchangeN(addr s.lock, true, AtomicAcquire)

proc release(s: var SpinLock) =
  atomicStoreN(addr s.lock, false, AtomicRelease)

template withLock(a: SpinLock, body: untyped) =
  acquire(a)
  try:
    body
  finally:
    release(a)

# Do stupid stuff
const
  numIters = 1_000
  numThreads = 1_000

var
  sum = 0
  threads: array[numThreads, Thread[SpinLock]]

proc routine(L: SpinLock) {.thread.} =
  var L = L
  for i in 1 .. numIters:
    withLock L:
      sum += i

# Driver code
proc main() =
  var L: SpinLock
  for i in 0..<numThreads:
    createThread(threads[i], routine, L)
  joinThreads(threads)
  echo sum

main()
