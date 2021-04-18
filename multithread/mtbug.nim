import locks
var a = 0
var
  lock: Lock
  threads: array[2, Thread[void]]

proc foo1() {.thread.} =
  var aLocal = 0
  for i in 0 ..< 500_000:
    aLocal = aLocal + i
  withLock lock:
    a = a + aLocal

proc foo2() {.thread.} =
  for i in 500_000 .. 1_000_000:
    withLock lock:
      a = a + i

proc main =
  initLock(lock)

  createThread(threads[0], foo1)
  createThread(threads[1], foo2)
  joinThread(threads[0])
  joinThread(threads[1])
  echo a

main()
