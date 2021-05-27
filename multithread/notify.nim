# https://github.com/google/sanitizers/wiki/ThreadSanitizerPopularDataRaces
# TSAN_OPTIONS="force_seq_cst_atomics=1"
import std/os

const
  delay = 1_000

var
  thread: Thread[void]
  proceed = false # Atomic
  bArrived = false

proc routine =
  var count = 0
  while true:
    if count mod delay == 0 and atomicLoadN(addr proceed, AtomicRelaxed):
      break
    cpuRelax()
    inc count
  doAssert bArrived

proc testNotify =
  createThread(thread, routine)
  sleep 10
  bArrived = true
  atomicStoreN(addr proceed, true, AtomicRelaxed)
  joinThread thread

testNotify()
