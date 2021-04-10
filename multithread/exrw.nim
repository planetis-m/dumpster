import std / os, rw2, benchtool, pbarrier, strformat, times

const
  numThreads = 5
  maxIter = 100_000

var
  m: RwMonitor # global object of monitor class
  b: Barrier
  readers: array[numThreads, Thread[int]]
  writers: array[numThreads, Thread[int]]
  fuel: int

proc gauge(id: int) =
  # each reader attempts to read 5 times
  var localFuel = 0
  wait b
  #echo "reader continues"
  for i in 0 ..< maxIter:
    readWith m:
      localFuel = fuel
      #echo "#", id, " observed fuel. Now left: ", fuel
    #sleep(500)
  wait b
  #echo "reader exits"
  echo localFuel

proc pump(id: int) =
  # each writer attempts to write 5 times
  wait b
  #echo "writer continues"
  for i in 0 ..< maxIter:
    writeWith m:
      #echo "#", id, " filled with fuel..."
      fuel += 30
      #sleep(250)
    #sleep(250)
  #echo "writer exits"
  wait b

proc main =
  initRwMonitor(m)
  initBarrier b, numThreads*2+1
  for i in 0 ..< numThreads:
    # creating threads which execute writer function
    createThread(writers[i], pump, i)
    # creating threads which execute reader function
    createThread(readers[i], gauge, i)
  warmup()
  #echo "bench starting"
  wait b
  let globalStart = epochTime()
  wait b
  echo "bench finished"
  let globalDuration = epochTime() - globalStart
  echo &"Time {globalDuration:>4.4f} s"
  joinThreads(readers)
  joinThreads(writers)
  assert fuel == 750
  destroyRwMonitor(m)

main()
