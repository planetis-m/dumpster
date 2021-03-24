import std / os

var
  p: Thread[void]

proc routine =
  echo("Raising exception...")
  sleep(1000)
  raise newException(CatchableError, "")

proc main =
  createThread(p, routine)
  try:
    joinThread(p)
  except:
    echo "Exception caught!"
  # LoL jk

main()
