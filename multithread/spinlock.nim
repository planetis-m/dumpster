# https://rigtorp.se/spinlock/
type
  SpinLock* = object
    lock: bool

proc acquire*(s: var SpinLock) =
  while true:
    # Optimistically assume the lock is free on the first try.
    if not atomicExchangeN(addr s.lock, true, AtomicAcquire)
      return
    else:
      # Wait for lock to be released without generating cache misses.
      while atomicLoadN(addr s.lock, AtomicRelaxed):
        # Reduces contention between hyper-threads.
        cpuRelax()

proc tryAcquire*(s: var SpinLock): bool =
  var oldLock = false
  # First do a relaxed load to check if lock is free in order to prevent
  # unnecessary cache misses if someone does `while not tryAcquire(s)`
  result = not atomicLoadN(addr s.lock, AtomicRelaxed) and
      not atomicExchangeN(addr s.lock, true, AtomicAcquire)

proc release*(s: var SpinLock) =
  atomicStoreN(addr s.lock, false, AtomicRelease)

template withSpinLock*(a: SpinLock, body: untyped) =
  acquire(a)
  try:
    body
  finally:
    release(a)
