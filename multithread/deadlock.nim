# https://www.youtube.com/watch?v=o0i2fc0Keo8
import std / [locks, os]

type
  CriticalData = ref object
    lock: Lock

  ThreadData = tuple
    id: int
    c1, c2: CriticalData

var
  threads: array[2, Thread[ThreadData]]

proc deadlock(data: ThreadData) =
  acquire data.c1.lock
  echo "Thread: ", data.id, " locking of the first mutex"
  sleep(1)
  acquire data.c2.lock
  echo "Thread: ", data.id, " locking of the second mutex"
  release data.c2.lock
  release data.c1.lock
  echo "Thread: ", data.id, " locking them both atomically"

proc newCriticalData: CriticalData =
  result = CriticalData()
  initLock(result.lock)

proc main =
  let c1 = newCriticalData()
  let c2 = newCriticalData()
  createThread(threads[0], deadlock, (id: 0, c1: c1, c2: c2))
  createThread(threads[1], deadlock, (id: 1, c1: c2, c2: c1))
  joinThread(threads[0])
  joinThread(threads[1])
  deinitLock(c1.lock)
  deinitLock(c2.lock)

main()
