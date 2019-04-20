import random, times, stats, strutils

template ff(f): untyped = formatFloat(f, ffDecimal, 3)

template printStats(name, acc, stats, dur): untyped =
   echo "\n", name,
      "\nCollected ", stats.n, " samples in ", dur.ff, " seconds",
      "\nAverage time: ", ff(stats.mean * 1000), " ms",
      "\nStddev  time: ", ff(stats.standardDeviationS * 1000), " ms",
      "\nMin     time: ", ff(stats.min * 1000), " ms",
      "\nMax     time: ", ff(stats.max * 1000), " ms",
      "\nDisplay accumulator to make sure it's not optimized away: ", acc

template benchAcc*(name: string; samples: int; code: untyped) =
   proc runBench() {.gensym.} =
      var stats: RunningStat
      let globalStart = cpuTime()
      var acc: typeof(code)
      for i in 1 .. samples:
         let start = cpuTime()
         acc = code
         let duration = cpuTime() - start
         stats.push duration
      let globalDuration = cpuTime() - globalStart
      printStats(name, acc, stats, globalDuration)
   runBench()
