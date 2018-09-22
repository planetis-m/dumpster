import math

const
   lookupSize = 4096
   sigmoidDomMin = -15.0
   sigmoidDomMax = 15.0
   interval = lookupSize.float / (sigmoidDomMax - sigmoidDomMin)

proc sigmoid(a: float): float {.inline.} =
   if a < -45.0: return 0
   if a > 45.0: return 1
   result = 1 / (1 + exp(-a))

proc createSigmoidTable(): array[lookupSize, float] =
   let f = (sigmoidDomMax - sigmoidDomMin) / lookupSize.float
   for i in 0 ..< lookupSize:
      result[i] = sigmoid(sigmoidDomMin + f * i.float)

const sigmoidTable = createSigmoidTable()

proc sigmoidCached(a: float): float {.inline.} =
   assert(a.classify != fcNaN)
   if a < sigmoidDomMin: return sigmoidTable[0]
   if a >= sigmoidDomMax: return sigmoidTable[lookupSize - 1]
   let j = int((a - sigmoidDomMin) * interval + 0.5)
   # Because floating point...
   if unlikely(j < 0): return sigmoidTable[0]
   if unlikely(j >= lookupSize): return sigmoidTable[lookupSize - 1]
   sigmoidTable[j]

when isMainModule:
   import times, strutils, random

   var xs, ys {.noInit.}: array[100, float]

   for i in 0 ..< xs.len:
      # Sets points from -15.0 to 15.0.
      xs[i] = rand(-15.0 .. 15.0)

   template bench*(name: string, code: untyped) =
      proc runBench() {.gensym.} =
         let start = epochTime()
         code
         let duration = epochTime() - start
         let timeStr = formatFloat(duration, ffDecimal, 3)
         echo name, ": ", timeStr
      runBench()

   bench("sigmoid"):
      for re in 0 ..< 10_000_000:
         for i in 0 ..< xs.len:
            ys[i] = sigmoid(xs[i])

   bench("sigmoidCached"):
      for re in 0 ..< 10_000_000:
         for i in 0 ..< xs.len:
            ys[i] = sigmoidCached(xs[i])
