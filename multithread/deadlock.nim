# https://www.youtube.com/watch?v=o0i2fc0Keo8
import std / [locks, os], fusion/smartptrs

type
  CriticalData = object
    L: Lock
    data {.guard: L.}: string

  ThreadData = tuple
    c1, c2: SharedPtr[CriticalData]

var
  threads: array[2, Thread[ThreadData]]

proc deadlock(data: ThreadData) =
  acquire data.c1[].L
  echo "Thread: ", getThreadId(), " locking of the first mutex"
  sleep 1
  acquire data.c2[].L
  echo "Thread: ", getThreadId(), " locking of the second mutex"
  release data.c2[].L
  release data.c1[].L
  echo "Thread: ", getThreadId(), " locking them both atomically"

proc newCriticalData: SharedPtr[CriticalData] =
  result = newSharedPtr(CriticalData())
  initLock(result[].L)

proc main =
  let c1 = newCriticalData()
  let c2 = newCriticalData()

  createThread(threads[0], deadlock, (c1: c1, c2: c2))
  createThread(threads[1], deadlock, (c1: c2, c2: c1))
  joinThread(threads[0])
  joinThread(threads[1])

  deinitLock(c1[].L)
  deinitLock(c2[].L)

main()
