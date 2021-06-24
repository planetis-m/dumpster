import std / os, thrsync, benchtool

const
  numThreads = 3
  maxIter = 100_000

var
  p: array[numThreads, Thread[int]]
  barrier: Barrier
  state: int

proc routine(i: int) =
  while atomicLoadN(addr state, AtomicAcquire) == 0:
    cpuRelax()
  #echo "thread continues"
  for i in 1 .. maxIter:
    #sleep(1000)
    #echo("Waiting at the barrier ", i)
    wait barrier
    #echo("Passed the barrier ", i)

proc main =
  initBarrier(barrier, numThreads+1)
  atomicStoreN(addr state, 0, AtomicRelease)
  for i in 0 ..< numThreads:
    createThread(p[i], routine, i)
  warmup()
  #echo "bench starting"
  atomicStoreN(addr state, 1, AtomicRelease)
  bench "reusable barrier", maxIter:
    wait barrier
  # another barrier here?
  joinThreads(p)

main()
