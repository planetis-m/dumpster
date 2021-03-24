import locks, os

const
  N = 6

var
  p: array[N, Thread[int]]
  mutex: Lock
  fuel: int
  readers: int
  readMutex: Lock

proc counter(i: int) =
  for _ in 0 ..< 5:
    acquire(readMutex)
    readers.inc
    if readers == 1:
      acquire(mutex)
    release(readMutex)

    echo "#", i, " observed fuel. Now left: ", fuel

    acquire(readMutex)
    readers.dec
    if readers == 0:
      release(mutex)
    release(readMutex)

    sleep(1000)

proc refuel(i: int) =
  for _ in 0 ..< 5:
    acquire(mutex)
    echo "#", i, " filled with fuel..."
    fuel += 30
    release(mutex)
    sleep(2000)

proc main =
  #randomize()
  initLock mutex
  initLock readMutex

  for i in 0 ..< N:
    if i == N-2 or i == N-1:
      createThread(p[i], refuel, i)
    else: createThread(p[i], counter, i)

  joinThreads(p)

  deinitLock readMutex
  deinitLock mutex

main()
