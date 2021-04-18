# https://www.youtube.com/watch?v=UrTU7ss3LDc
import std / [locks, os, random]

const
  numThreads = 10

var
  stoveMutex: array[4, Lock]
  stoveFuel: array[4, int] = [100, 100, 100, 100]
  p: array[numThreads, Thread[int]]

proc routine(id: int) =
  var i = 0
  while true:
    if tryAcquire(stoveMutex[i]):
      let fuelNeeded = rand(30)
      if stoveFuel[i] - fuelNeeded < 0:
        echo(id, " No more fuel... going home")
      else:
        stoveFuel[i] -= fuelNeeded
        sleep(1000)
        echo(id, " Fuel left ", stoveFuel[i])
      release(stoveMutex[i])
      break
    else:
      if i == stoveMutex.high:
        echo(id, " No stove available yet, waiting...")
        sleep(3000)
        i = 0
      else: i.inc

proc main =
  #randomize()
  for i in 0 .. stoveMutex.high:
    initLock stoveMutex[i]
  for i in 0 ..< numThreads:
    createThread(p[i], routine, i)
  joinThreads(p)
  for i in 0 .. stoveMutex.high:
    deinitLock stoveMutex[i]

main()
