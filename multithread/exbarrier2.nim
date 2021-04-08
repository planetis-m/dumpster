import std / os, pbarrier, benchtool

const
  numThreads = 3
  maxIter = 10_000

var
  p: array[numThreads, Thread[int]]
  barrier: SyncBarrier
  benchStart: SyncBarrier

proc routine(i: int) =
  discard wait benchStart
  #echo "thread continues"
  for i in 1 .. maxIter:
    #sleep(1000)
    #echo("Waiting at the barrier ", i)
    discard wait barrier
    #echo("Passed the barrier ", i)

proc main =
  init(barrier, numThreads+1)
  init(benchStart, numThreads+1)
  for i in 0 ..< numThreads:
    createThread(p[i], routine, i)

  warmup()
  #echo "bench starting"
  discard wait benchStart
  bench "pthread barrier", maxIter:
    discard wait barrier
  joinThreads(p)
  delete benchStart
  delete barrier

main()
