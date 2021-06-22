import std/[os, isolation, atomics]

var
  p: Thread[void]
  exc: Isolated[ref Exception]
  excIsSet: Atomic[bool]

proc exceptHook(e: sink Isolated[ref Exception]) =
  # Only one's thread exception will be propagated to the main.
  if not excIsSet.load(moRelaxed) and
      not excIsSet.exchange(true, moAcquire):
    exc = e

proc routine {.thread.} =
  try:
    echo("Raising exception...")
    sleep(100)
    raise newException(CatchableError, "From thread: " & $getThreadId())
  except:
    exceptHook(isolate(getCurrentException()))

proc main =
  createThread(p, routine)
  try:
    joinThread(p)
    if excIsSet.load(moAcquire):
      raise exc.extract
  except:
    echo "Exception caught! ", getCurrentExceptionMsg()

main()
