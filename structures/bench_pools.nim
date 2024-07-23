import std/[times, stats, strformat, random]
import pools_destroy

type
  Matrix = object
    m0, m4, m8, m12: float
    m1, m5, m9, m13: float
    m2, m6, m10, m14: float
    m3, m7, m11, m15: float

# proc `=destroy`(v: Matrix) =
#   discard

const
  DefaultAlignment = 8

proc runThroughputBenchmark(duration: float, poolSize: int) =
  var backingBuffer {.align: DefaultAlignment.}: array[1024 * 1024, byte]
  var pool: FixedPool[Matrix]
  init(pool, backingBuffer)

  var ptrs = newSeq[ptr Matrix](poolSize)
  var allocated = newSeq[bool](poolSize)
  var operationCount: int = 0
  var throughputStats: RunningStat

  let endTime = cpuTime() + duration
  while cpuTime() < endTime:
    let startTime = cpuTime()

    # Perform a mix of allocations and deallocations
    for _ in 0 ..< 1000:  # Batch size
      let index = rand(poolSize - 1)
      if not allocated[index]:
        ptrs[index] = alloc(pool)
        allocated[index] = true
      else:
        dealloc(pool, ptrs[index])
        allocated[index] = false
      inc operationCount

    let elapsedTime = cpuTime() - startTime
    let batchThroughput = 1000.0 / elapsedTime  # operations per second
    throughputStats.push(batchThroughput)

  let totalTime = duration
  let overallThroughput = float(operationCount) / totalTime

  echo &"""
Throughput Benchmark Results (Pool Size: {poolSize}):
  Total Operations: {operationCount}
  Overall Throughput: {overallThroughput:.2f} ops/sec
  Mean Batch Throughput: {throughputStats.mean:.2f} ops/sec
  Min Batch Throughput: {throughputStats.min:.2f} ops/sec
  Max Batch Throughput: {throughputStats.max:.2f} ops/sec
  Throughput Std Dev: {throughputStats.standardDeviation:.2f} ops/sec
"""

proc main() =
  echo "Fixed Pool Allocator Throughput Benchmark"
  echo "========================================="

  randomize()  # Initialize random number generator
  let duration = 5.0  # Run each benchmark for 5 seconds
  let poolSizes = [100, 1000, 10000]

  for size in poolSizes:
    runThroughputBenchmark(duration, size)

main()
