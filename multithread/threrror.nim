import std/[os, isolation, atomics]

var
  p: Thread[void]
  exc: ref Exception
  excIsSet: Atomic[bool]

proc exceptHook(e: sink Isolated[ref Exception]) {.nodestroy.} =
  # Only one's thread exception will be propagated to the main.
  if not excIsSet.load(moRelaxed) and
      not excIsSet.exchange(true, moAcquire):
    exc = e.extract
  else: `=destroy`(e)

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
      raise exc
  except:
    echo "Exception caught! ", getCurrentExceptionMsg()
    wasMoved(exc)

main()
