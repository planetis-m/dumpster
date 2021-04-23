# https://www.youtube.com/watch?v=ZQFzMfHIxng&t=1s
import std / locks

const
  numIters = 1_000
  numThreads = 1_000

var
  sum = 0
  mutex: Lock
  threads: array[numThreads, Thread[void]]

proc doWork1() {.thread.} =
  for i in 1 .. numIters:
    #var tmp = sum
    #tmp += i
    #stdout.write tmp, " "
    #sum = tmp
    stdout.write atomicFetchAdd(addr sum, i, AtomicRelaxed), " "

proc doWork2() {.thread.} =
  var s = 0
  for i in 1 .. numIters:
    s += i
  acquire mutex
  sum += s
  release mutex

# Driver code
proc main() =
  initLock(mutex)
  for i in 0..<numThreads:
    createThread(threads[i], doWork1)
  joinThreads(threads)
  echo sum
  deinitLock(mutex)

main()
