# https://rigtorp.se/split-locks/
import times

proc main =
  var buf: array[128, char]
  var v = cast[ptr uint32](cast[uint64](buf) or 61)

  let start = epochTime()
  const numIters = 1_000_000
  for i in 0 ..< numIters:
    discard atomicFetchAdd(v, 1, AtomicAcquire)
  let stop = epochTime()
  echo (stop - start) / numIters.float

main()
