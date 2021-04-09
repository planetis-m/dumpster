import os, thrsync

const
  N = 2

var
  aThread, bThread: Thread[void]
  aArrived, bArrived: Semaphore

proc a =
  echo "A starts"
  signal aArrived
  blockUntil bArrived
  sleep(1000)
  echo "A progresses"

proc b =
  echo "B starts"
  signal bArrived
  blockUntil aArrived
  sleep(2000)
  echo "B progresses"

proc main =
  #randomize()
  initSemaphore aArrived
  initSemaphore bArrived

  createThread(aThread, b)
  createThread(bThread, a)
  joinThread(aThread)
  joinThread(bThread)

  destroySemaphore aArrived
  destroySemaphore bArrived

main()
