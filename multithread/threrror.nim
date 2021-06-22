import std/[os, isolation, atomics]

const
  numThreads = 4

var
  threads: array[numThreads, Thread[void]]
  exc: Isolated[ref Exception]
  excIsSet: Atomic[bool]

proc exceptHook(e: sink Isolated[ref Exception]) =
  # Only one's thread exception will be propagated to the main.
  if not excIsSet.load(moRelaxed) and
      not exchange(excIsSet, true, moRelaxed):
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
  joinThreads(threads)
  try:
    if excIsSet.load(moRelaxed):
      raise exc.extract
  except:
    echo "Exception caught! ", getCurrentExceptionMsg()

main()
