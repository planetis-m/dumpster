# https://www.youtube.com/watch?v=emYmQQZXRT8
# Godbolt compiler options: --threads:on -d:danger --gc:arc --panics:on -g -t:"-O3"
var
  line = 0
  now_serving {.volatile.} = 0
  shared_value = 0

proc compute: int {.noinline.} = discard

proc memory_rerorder {.exportc.} =
  # Wait for the lock
  let ticket = atomicFetchAdd(addr line, 1, AtomicSeqCst)
  while now_serving != ticket: discard
  # Critical Section
  shared_value = compute()
  # SW memory barrier
  {.emit: """asm volatile("" ::: "memory");""".}
  # Release the lock
  now_serving = ticket + 1
