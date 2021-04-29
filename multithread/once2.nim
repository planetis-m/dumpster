
type
  Once*[T] = object
    L: Lock # ???
    val: ptr T

proc `=destroy`*[T](p: var Once[T]) =
  let p = atomicLoadN(addr o.val, AtomicAcquire)
  if p != nil:
    acquire(o.L)
    p = atomicLoadN(addr o.val, AtomicRelaxed)
    if p != nil:
      deallocShared(p)
      atomicStoreN(addr o.val, nil, AtomicRelease) # ??
    release(o.L)
    # deinitLock(o.L) ??

proc `=copy`*[T](dest: var Once[T], src: Once[T]) {.error.}

proc initOnce*[T](o: var Once[T]) = # ??
  initLock(o.L)

proc `[]`*[T](o: Once[T]): var T {.nodestroy.} =
  result = atomicLoadN(addr o.val, AtomicAcquire)
  if result == nil:
    acquire(o.L)
    result = atomicLoadN(addr o.val, AtomicRelaxed)
    if result == nil:
      result = cast[typeof(result.val)](allocShared(sizeof(result.val[])))
      atomicStoreN(addr o.val, result, AtomicRelease)
    release(o.L)
