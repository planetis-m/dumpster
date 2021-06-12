# --threads:on --threadanalysis:off
import std/[locks, typetraits, os]

type
  LockGuard = distinct ptr Lock

proc `=destroy`(x: var LockGuard) =
  release(distinctBase(x)[])

proc lock(x: var Lock): LockGuard =
  acquire(x)
  result = LockGuard(addr x)

const
  numThreads = 4
  numIters = 10

var
  L: Lock
  data: string
  threads: array[numThreads, Thread[void]]

proc routine {.thread.} =
  for i in 0..<numIters:
    var L: LockGuard
    #discard lock(L) # It works! ...but an optimization might kill it
    let tmp = data
    data = "abc"
    sleep 1
    data = tmp & "f"

proc frob =
  initLock L
  for i in 0..<numThreads:
    createThread(threads[i], routine)
  for i in 0..<numIters:
    let g {.used.} = lock(L) # also really annoying cause unused variable
    doAssert data != "abc"
  joinThreads(threads)

  let g = lock(L)
  doAssert data.len == 40

frob()
