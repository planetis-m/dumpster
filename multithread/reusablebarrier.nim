#https://www.youtube.com/watch?v=xkpNYZIPTR4
import std / os, nlocks

const
  threadNum = 3

var
  p: array[threadNum, Thread[int]]
  releaseCond: Cond
  counterLock: Lock
  counter: int
  globalSense: bool

proc routine(i: int) =
  var localSense = globalSense
  while true:
    localSense = not localSense
    echo("Waiting at the barrier ", i)

    acquire counterLock
    counter.inc
    if counter == threadNum:
      counter = 0
      globalSense = localSense
      broadcast releaseCond
    else:
      #while globalSense != localSense:
      wait(releaseCond, counterLock)
      assert globalSense == localSense
    release counterLock

    echo("Passed the barrier ", i)
    #sleep(1000)

proc main =
  initCond(releaseCond)
  initLock(counterLock)

  for i in 0 ..< threadNum:
    createThread(p[i], routine, i)
  joinThreads(p)

  deinitCond(releaseCond)
  deinitLock(counterLock)

main()
