# Invalid don't look
import std/[os, isolation, atomics]

const
  numThreads = 4

var
  threads: array[numThreads, Thread[void]]
  exc: ref Exception
  excIsSet: Atomic[bool]

proc exceptHook(e: sink Isolated[ref Exception]) =
  # Only one's thread exception will be propagated to the main.
  if not excIsSet.load(moRelaxed) and
      not exchange(excIsSet, true, moRelaxed):
    exc = extract e

proc routine {.thread.} =
  try:
    echo("Raising exception... ", getThreadId())
    sleep(100)
    raise newException(CatchableError, "From thread: " & $getThreadId())
  except:
    exceptHook(unsafeIsolate(getCurrentException()))

proc main =
  for i in 0..<numThreads:
    createThread(threads[i], routine)
  joinThreads(threads)
  try:
    if excIsSet.load(moRelaxed): # no need for atomic!
      raise exc.extract
  except:
    echo "Exception caught! ", getCurrentExceptionMsg()

main()
