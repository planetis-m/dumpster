import math

proc hypot2(a, b: float): float =
   # sqrt(a^2 + b^2) without under/overflow.
   if abs(a) > abs(b):
      result = b / a
      result = abs(a) * sqrt(1 + result * result)
   elif b != 0:
      result = a / b
      result = abs(b) * sqrt(1 + result * result)
   else:
      result = 0.0

when isMainModule:
   import random, times, strutils

   const samples = 10_000_000
   var values = newSeq[float](2 * samples)
   for i in 0 .. high(values):
      values[i] = rand(-100.0 .. 100.0)

   block benchHypot:
      var sum = 0.0
      let start = epochTime()
      for i in countup(0, high(values), 2):
         sum += hypot2(values[i], values[i + 1])

      let duration = epochTime() - start
      echo formatFloat(duration, ffDecimal, 3), "us --- s:", sum

   block benchSqrt:
      var sum = 0.0
      let start = epochTime()
      for i in countup(0, high(values), 2):
         sum += sqrt(values[i] * values[i] + values[i + 1] * values[i + 1])

      let duration = epochTime() - start
      echo formatFloat(duration, ffDecimal, 3), "us --- s:", sum
