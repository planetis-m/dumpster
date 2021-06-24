import sync, std/[os, locks]

const
  N = 6

var
  p: array[N, Thread[int]]
  sem: Semaphore
  fuel: int
  readers: int
  readMutex: Lock

proc counter(i: int) =
  for _ in 0 ..< 5:
    acquire(readMutex)
    readers.inc
    if readers == 1:
      wait(sem)
    release(readMutex)

    echo "#", i, " observed fuel. Now left: ", fuel

    acquire(readMutex)
    readers.dec
    if readers == 0:
      signal(sem)
    release(readMutex)

    sleep(1000)

proc refuel(i: int) =
  for _ in 0 ..< 5:
    wait(sem)
    echo "#", i, " filled with fuel..."
    fuel += 30
    signal(sem)
    sleep(2000)

proc main =
  #randomize()
  initSemaphore sem, 1
  initLock readMutex

  for i in 0 ..< N:
    if i == N-2 or i == N-1:
      createThread(p[i], refuel, i)
    else: createThread(p[i], counter, i)

  joinThreads(p)
  assert fuel == 300

  deinitLock readMutex

main()
