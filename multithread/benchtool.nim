import times, stats, strformat

proc warmup*() =
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

template bench*(name, samples, code: untyped) =
  var stats: RunningStat
  let globalStart = cpuTime()
  for i in 0 ..< samples:
    let start = cpuTime()
    code
    let duration = cpuTime() - start
    stats.push duration
  let globalDuration = cpuTime() - globalStart
  printStats(name, stats, globalDuration)
