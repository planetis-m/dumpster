import std / [os, tables], thrsync

const
  numThreads = 5

var
  m: RwMonitor # global object of monitor class
  readers: array[numThreads, Thread[int]]
  writers: array[numThreads, Thread[int]]
  ct: CountTable[string]

proc read(id: int) {.thread.} =
  # each reader attempts to read 10 times
  for i in 0 ..< 10:
    readWith m:
      echo "#", id, " reads table: ", ct
    sleep(500)

proc write(id: int) {.thread.} =
  # each writer attempts to write 5 times
  for i in 0 ..< 5:
    writeWith m:
      echo "#", id, " writes to table..."
      ct.inc($id)
      sleep(250)
    sleep(250)

proc toCountTable[A](pairs: openArray[(A, int)]): CountTable[A] =
  result = initCountTable[A](pairs.len)
  for key, val in pairs.items: result.inc(key, val)

proc main =
  initRwMonitor(m)
  for i in 0 ..< numThreads:
    # creating threads which execute writer function
    createThread(writers[i], write, i)
    # creating threads which execute reader function
    createThread(readers[i], read, i)
  joinThreads(readers)
  joinThreads(writers)
  doAssert ct == {"0": 5, "1": 5, "2": 5, "3": 5, "4": 5}.toCountTable
  destroyRwMonitor(m)

main()
