# github: RichardKnop/ansi-c-perceptron
import math, random, strscans, strformat
# -O2 -ftree-vectorize -fopt-info-vec-missed
# -Wno-discarded-qualifiers -fopt-info
when defined(debugAsm):
   {.passC: "-fverbose-asm -masm=intel -S".}
when defined(fastmath):
   {.passC: "-ffast-math".}
when defined(marchNative):
   {.passC: "-march=native".}

const
   learningRate = 0.1
   maxIteration = 100
   halfOfIter = maxIteration div 2
   threeFourthsOfIter = (maxIteration div 4) * 3

proc calculateOutput(weights: array[3, float]; x, y: float): int {.inline.} =
   let sum = x * weights[0] + y * weights[1] + weights[2]
   if sum >= 0: 1 else: -1

proc main() =
   randomize(5155) # reproducible
   var
      fp: File
      x: array[208, float]
      y: array[208, float]
      weights: array[3, float]
      outputs: array[208, int]
   if not fp.open("test1.txt", fmRead):
      quit("Cannot open file.")
   var i = 0
   for s in fp.lines:
      doAssert scanf(s, "$f$s$f$s$i", x[i], y[i], outputs[i])
      if outputs[i] == 0:
         outputs[i] = -1
      i.inc
   fp.close
   let patternCount = i
   weights[0] = rand(1.0)
   weights[1] = rand(1.0)
   weights[2] = rand(1.0)
   var iter = 0
   var eps = pow(2.0, -966.0)
   while true:
      iter.inc
      if iter == halfOfIter or iter == threeFourthsOfIter:
         echo("Perceptron taking a long time: making convergence criterion less exact.")
         eps = pow(0.8, eps)
      var globalError = 0.0
      for p in 0 ..< patternCount:
         let output = calculateOutput(weights, x[p], y[p])
         let localError = float(outputs[p] - output)
         weights[0] += learningRate * localError * x[p]
         weights[1] += learningRate * localError * y[p]
         weights[2] += learningRate * localError
         globalError += localError * localError
      # Root Mean Squared Error
      echo(&"Iteration {iter} : RMSE = {sqrt(globalError / patternCount.float):.4f}")
      if globalError <= eps or iter >= maxIteration:
         break
   echo(&"\nDecision boundary (line) equation: {weights[0]:.2f}*x + {weights[1]:.2f}*y + {weights[2]:.2f} = 0")

main()
