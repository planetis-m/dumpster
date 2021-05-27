import std/[os, strformat, volatile]
const
  numThreads = 10
  numIters = 100

var
  threads: array[numThreads, Thread[int]]
  start {.volatile.}: array[numThreads, int]
  ended {.volatile.}: array[numThreads, int]

proc routine(id: int) =
  while volatileLoad(addr start[id]) == 0: fence() # Wait until work started
  echo &"Thread {id} started"
  volatileStore addr ended[id], 1 # Indicate that work completed
  echo &"Thread {id} finished"

proc dowork =
  for i in 0..<numThreads:
    volatileStore addr start[i], 1 # Start thread working
  fence()
  for i in 0..<numThreads:
    while volatileLoad(addr ended[i]) == 1: fence() # Wait until thread completes work

proc testNotify =
  for i in 0..<numThreads:
    createThread(threads[i], routine, i)
  dowork()
  joinThreads threads

testNotify()
