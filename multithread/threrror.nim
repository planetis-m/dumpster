import std/[os, isolation]

const
  numThreads = 4

var
  threads: array[numThreads, Thread[void]]
  exc: Isolated[ref Exception]
  excIsSet: bool

proc exceptHook(e: sink Isolated[ref Exception]) =
  # Only one's thread exception will be propagated to the main.
  if not atomicLoadN(addr excIsSet, AtomicRelaxed) and
      not atomicExchangeN(addr excIsSet, true, AtomicAcquire):
    exc = e

proc routine {.thread.} =
  try:
    echo("Raising exception... ", getThreadId())
    sleep(100)
    raise newException(CatchableError, "From thread: " & $getThreadId())
  except:
    exceptHook(isolate(getCurrentException()))

proc main =
  for i in 0..<numThreads:
    createThread(threads[i], routine)
  try:
    joinThreads(threads)
    if excIsSet: # no need for atomic!
      raise exc.extract
  except:
    echo "Exception caught! ", getCurrentExceptionMsg()

main()
