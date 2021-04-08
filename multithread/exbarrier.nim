import std / os, pbarrier2, benchtool

const
  numThreads = 3
  maxIter = 100_000

var
  p: array[numThreads, Thread[int]]
  barrier: Barrier
  benchStart: Barrier

proc routine(i: int) =
  wait benchStart
  #echo "thread continues"
  for i in 1 .. maxIter:
    #sleep(1000)
    #echo("Waiting at the barrier ", i)
    wait barrier
    #echo("Passed the barrier ", i)

proc main =
  initBarrier(barrier, numThreads+1)
  initBarrier(benchStart, numThreads+1)
  for i in 0 ..< numThreads:
    createThread(p[i], routine, i)
  warmup()
  #echo "bench starting"
  wait benchStart
  bench "reusable barrier", maxIter:
    wait barrier
  joinThreads(p)
  destroyBarrier benchStart
  destroyBarrier barrier

main()
