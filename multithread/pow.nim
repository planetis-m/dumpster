# https://blog.softwaremill.com/multithreading-in-rust-with-mpsc-multi-producer-single-consumer-channels-db0fc91ae3fa
import std/[sha1, strutils, strformat], sync

const
  Base = 42
  numThreads = 4
  Backoff = 10
  Delay = 1_000

type
  Solution = object
    number: int
    hash: Sha1Digest

var
  isSolutionFound: Semaphore
  shutdown = false
  threads: array[numThreads, Thread[int]]
  queue: MpscQueue[Solution]

func verifyNumber(number: int; solution: var Solution): bool =
  let hash = secureHash($(number * Base)).Sha1Digest
  for i in hash.len - 3 .. hash.len - 1:
    if hash[i] != 0'u8:
      return false
  solution.number = number
  solution.hash = hash
  result = true

proc searchForSolution(startAt: int) =
  var
    solution: Solution
    number = startAt
  while not shutdown:
    if verifyNumber(number, solution):
      signal isSolutionFound
      queue.enqueue(solution)
      return
    inc number, numThreads

proc main =
  echo(&"Attempting to find a number, which - while multiplied by {Base} and hashed using SHA1 - will result in a hash ending with 000000. \nPlease wait...")
  queue = newMpscQueue[Solution]()
  initSemaphore isSolutionFound
  for i in 0 ..< numThreads:
    createThread(threads[i], searchForSolution, i)

  blockUntil isSolutionFound
  shutdown = true
  var solution: Solution
  while not queue.dequeue(solution): cpuRelax()
  echo(&"Found the solution.\nThe number is: {solution.number}.\nResult hash: {SecureHash(solution.hash)}.")

  joinThreads(threads)

main()
