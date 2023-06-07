# github: RichardKnop/ansi-c-perceptron
import math, random, parsecsv, strformat, strutils
# -O2 -ftree-vectorize -fopt-info-vec-missed
# -Wno-discarded-qualifiers -fopt-info
when defined(debugAsm):
  {.passC: "-fverbose-asm -masm=intel -S".}
when defined(fastmath):
  {.passC: "-ffast-math".}
when defined(marchNative):
  {.passC: "-march=native".}

const
  learningRate = 0.0001
  maxIteration = 100

proc calculateOutput(weights: array[3, float]; x, y: float): int {.inline.} =
  let sum = x * weights[0] + y * weights[1] + weights[2]
  if sum >= 0: 1 else: -1

proc main() =
  randomize(5155) # reproducible
  var
    x: array[208, float]
    y: array[208, float]
    weights: array[3, float]
    outputs: array[208, int]
  var cp: CsvParser
  open(cp, "test1.txt", '\t')
  var i = 0
  while readRow(cp):
    x[i] = parseFloat(cp.row[0])
    y[i] = parseFloat(cp.row[1])
    outputs[i] = if parseInt(cp.row[2]) == 0: -1 else: 1
    i.inc
  close(cp)
  let patternCount = i
  for i in countdown(x.high, 1):
    let j = rand(i)
    swap(x[i], x[j])
    swap(y[i], y[j])
    swap(outputs[i], outputs[j])
  weights = [rand(1.0), rand(1.0), rand(1.0)]
  var iter = 0
  var eps = pow(2.0, -966.0)
  while true:
    iter.inc
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
