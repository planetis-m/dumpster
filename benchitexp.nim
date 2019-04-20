import random, times, stats, strutils

template ff(f): untyped = formatFloat(f, ffDecimal, 3)

proc warmup*() =
   # Warmup - make sure cpu is on max perf
   let start = cpuTime()
   var foo = 123
   for i in 0 ..< 300_000_000:
      foo += i * i mod 456
      foo = foo mod 789
   # Compiler shouldn't optimize away the results as cpuTime rely on sideeffects
   let duration = cpuTime() - start
   echo "Warmup: ", ff(duration), " s, result ", foo, " (displayed to avoid compiler optimizing warmup away)"

template printStats(name, stats, dur) =
   echo "\n", name,
      "\nCollected ", stats.n, " samples in ", ff(dur), " seconds",
      "\nAverage time: ", ff(stats.mean * 1000), " ms",
      "\nStddev  time: ", ff(stats.standardDeviationS * 1000), " ms",
      "\nMin     time: ", ff(stats.min * 1000), " ms",
      "\nMax     time: ", ff(stats.max * 1000), " ms"

template bench*(name: string; samples: int; code: untyped) =
   proc runBench() {.gensym.} =
      var stats: RunningStat
      let globalStart = cpuTime()
      for i in 1 .. samples:
         let start = cpuTime()
         code
         let duration = cpuTime() - start
         stats.push duration
      let globalDuration = cpuTime() - globalStart
      printStats(name, stats, globalDuration)
   runBench()

template benchIt*(name: string; typ: type; samples: int; code: untyped) =
   proc runBench() {.gensym.} =
      var stats: RunningStat
      let globalStart = cpuTime()
      var it {.inject.}: typ
      for i in 1 .. samples:
         let start = cpuTime()
         code
         let duration = cpuTime() - start
         stats.push duration
      let globalDuration = cpuTime() - globalStart
      printStats(name, stats, globalDuration)
      echo "\nDisplay it to make sure it's not optimized away: ", it
   runBench()
