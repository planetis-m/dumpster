import locks, os

const
  N = 6

var
  p: array[N, Thread[int]]
  mutex: Lock
  fuelCond: Cond
  fuel: int

proc car(i: int) =
  acquire(mutex)
  while fuel < 40:
    echo "#", i, " no fuel. Waiting..."
    wait(fuelCond, mutex)
  fuel -= 40
  echo "#", i, " got fuel. Now left: ", fuel
  release(mutex)

proc refuel(i: int) =
  for _ in 0 ..< 5:
    acquire(mutex)
    echo "#", i, " filled with fuel..."
    fuel += 30
    signal(fuelCond)
    release(mutex)
    sleep(2000)

proc main =
  #randomize()
  initLock mutex
  initCond fuelCond

  for i in 0 ..< N:
    if i == N-2 or i == N-1:
      createThread(p[i], refuel, i)
    else: createThread(p[i], car, i)

  joinThreads(p)

  deinitCond fuelCond
  deinitLock mutex

main()
