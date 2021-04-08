import std / os, threadutils, benchtool

const
  numThreads = 3
  maxIter = 10_000

var
  p: array[numThreads, Thread[int]]
  barrier: Barrier
  benchStart: Barrier

proc routine(i: int) =
  sync benchStart
  #echo "thread continues"
  for i in 1 .. maxIter:
    #sleep(1000)
    #echo("Waiting at the barrier ", i)
    sync barrier
    #echo("Passed the barrier ", i)

proc main =
  initBarrier(barrier, numThreads+1)
  initBarrier(benchStart, numThreads+1)
  for i in 0 ..< numThreads:
    createThread(p[i], routine, i)

  warmup()
  #echo "bench starting"
  sync benchStart
  bench "reusable barrier", maxIter:
    sync barrier
  joinThreads(p)
  destroyBarrier benchStart
  destroyBarrier barrier

main()
