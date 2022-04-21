# Also use -Wno-discarded-qualifiers -fopt-info
when defined(debugAsm):
   {.passC: "-fverbose-asm -masm=intel -S".}
when defined(fastmath):
   {.passC: "-ffast-math".}
when defined(marchNative):
   {.passC: "-march=native".}

import random
import strutils
import std/[times, stats, strformat]

const MaxIter = 1_000_000

proc warmup() =
  # Warmup - make sure cpu is on max perf
  let start = cpuTime()
  var a = 123
  for i in 0 ..< 300_000_000:
    a += i * i mod 456
    a = a mod 789
  let dur = cpuTime() - start
  echo &"Warmup: {dur:>4.4f} s ", a

proc printStats(name: string, stats: RunningStat, dur: float) =
  echo &"""{name}:
  Collected {stats.n} samples in {dur:>4.4f} s
  Average time: {stats.mean * 1000:>4.4f} ms
  Stddev  time: {stats.standardDeviationS * 1000:>4.4f} ms
  Min     time: {stats.min * 1000:>4.4f} ms
  Max     time: {stats.max * 1000:>4.4f} ms"""

template bench(name, samples, code: untyped) =
  var stats: RunningStat
  let globalStart = cpuTime()
  for i in 0 ..< samples:
    let start = cpuTime()
    code
    let duration = cpuTime() - start
    stats.push duration
  let globalDuration = cpuTime() - globalStart
  printStats(name, stats, globalDuration)

const
   n = 16384 div sizeof(float32) # L1 DCache size

proc generateVector(n: int): seq[float32] =
   result = newSeq[float32](n)
   for i in 0 ..< n:
      result[i] = rand(1.0)

proc dotProduct(listA, listB: openarray[float32]): float32 =
   #
   # listA: a list of numbers
   # listB: a list of numbers of the same length as listA
   #
   let m = len(listA)
   for i in 0 ..< m:
      result += listA[i] * listB[i]

proc main() =
   #warmup()
   randomize(128)
   echo "Generating ", n, " element vectors."
   let
      listA = generateVector(n)
      listB = generateVector(n)
   var res = 0'f32
   bench("TBitSet", MaxIter):
      res = dotProduct(listA, listB)
   echo res

main()
