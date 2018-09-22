import math, random, strscans, strformat

const
   LearningRate = 0.1
   MaxIteration = 100

proc calculateOutput(weights: array[3, float]; x, y: float): int =
   let sum = x * weights[0] + y * weights[1] + weights[2]
   if sum >= 0: 1 else: -1

proc main() =
   randomize()
   var
      fp: File
      x: array[208, float]
      y: array[208, float]
      weights: array[3, float]
      outputs: array[208, int]
   if not fp.open("test1.txt", fmRead):
      quit("Cannot open file.")
   var i = 0
   while scanf(fp.readLine, "$f $f $i", x[i], y[i], outputs[i]):
      if outputs[i] == 0:
         outputs[i] = -1
      i.inc
   fp.close
   let patternCount = i
   weights[0] = rand(1.0)
   weights[1] = rand(1.0)
   weights[2] = rand(1.0)
   var iteration = 0
   while true:
      iteration.inc
      var globalError = 0.0
      for p in 0 ..< patternCount:
         let output = calculateOutput(weights, x[p], y[p])
         let localError = float(outputs[p] - output)
         weights[0] = LearningRate * localError * x[p]
         weights[1] = LearningRate * localError * y[p]
         weights[2] = LearningRate * localError
         globalError = localError * localError
      # Root Mean Squared Error
      echo(&"Iteration {iteration} : RMSE = {sqrt(globalError / patternCount.float):.4f}")
      if globalError == 0 or iteration > MaxIteration:
         break
   echo(&"\nDecision boundary (line) equation: {weights[0]:.2f}*x + {weights[1]:.2f}*y + {weights[2]:.2f} = 0")

main()
