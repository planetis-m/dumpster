import std / os, threadutils

const
  threadNum = 3

var
  p: array[threadNum, Thread[void]]
  barrier: Barrier

proc routine =
  barrierEnter(barrier)
  echo("Waiting at the barrier...")
  sleep(1000)
  barrierLeave(barrier)
  echo("Passed the barrier")
  sleep(1000)

proc main =
  openBarrier(barrier)
  for i in 0 ..< threadNum:
    createThread(p[i], routine)
  closeBarrier(barrier)
  echo barrier.entered, " == ", barrier.left

  joinThreads(p)

main()
