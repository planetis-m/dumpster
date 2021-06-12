import readerwriter, std/os

const
  numThreads = 4
  numIters = 10

var
  rw: RwMonitor[string]
  threads: array[numThreads, Thread[void]]

proc routine {.thread.} =
  for i in 0..<numIters:
    let w = write(rw)
    let tmp = w.val
    w.val = "abc"
    sleep 1
    w.val = tmp & "f"

proc frob =
  initRwMonitor rw, ""
  for i in 0..<numThreads:
    createThread(threads[i], routine)
  for i in 0..<numIters:
    let r = read(rw)
    assert r.val != "abc"
  joinThreads(threads)

  val.readWith(rw):
    assert val.len == 40

frob()
