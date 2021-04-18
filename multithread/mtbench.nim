import std / monotimes

var
  state: int
  threads: Thread[void]

const
  sleepMs = 20000
  iterations = 100

proc doWork =
  while atomicLoadN(addr state, AtomicAcquire) == 0:
    cpuRelax()
  while atomicLoadN(addr state, AtomicAcquire) == 1:
    discard

proc main =
  atomicStoreN(addr state, 0, AtomicRelease)
  for i in 0 ..< numThreads:
    createThread(threads[i], doWork)

  let start = getMonoTime()
  atomicStoreN(addr state, 1, AtomicRelease)
  sleep(sleepMs)
  atomicStoreN(addr state, 2, AtomicRelease)
  let duration = getMonoTime() - start

  joinThreads(threads)

main()
