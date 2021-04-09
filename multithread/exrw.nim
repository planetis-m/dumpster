import std / os, readerwriter

var
  m: RwMonitor # global object of monitor class
  readers: array[5, Thread[int]]
  writers: array[5, Thread[int]]
  fuel: int

proc reader(id: int) =
  # each reader attempts to read 5 times
  for i in 0 ..< 5:
    #sleep(1000)
    writer m:
      echo "#", id, " observed fuel. Now left: ", fuel
      #sleep(500)

proc writer(id: int) =
  # each writer attempts to write 5 times
  for i in 0 ..< 5:
    #sleep(1000)
    reader m:
      echo "#", id, " filled with fuel..."
      fuel += 30
      #sleep(1000)

proc main =
  initRwMonitor(m)
  for i in 0 ..< 5:
    # creating threads which execute reader function
    createThread(readers[i], reader, i)
    # creating threads which execute writer function
    createThread(writers[i], writer, i)
  joinThreads(readers)
  joinThreads(writers)
  echo fuel
  destroyRwMonitor(m)

main()
