#
# implements a version of the algorithm set out at
# http://natureofcode.com/book/chapter-10-neural-networks/ ,
# but without graphics
#
import random

type
   Inputs = array[0..2, int]
   Weights = array[0..2, float]

func targetOutput(a, b: int): int =
   # the function the perceptron will be learning is f(x) = 2x + 1
   result = if a * 2 + 1 < b: 1 else: -1

template write(x) =
   stdout.write(x)

proc showTargetOutput() =
   for y in countdown(10, -9):
      for x in -9 .. 10:
         if targetOutput(x, y) == 1:
            write('#')
         else:
            write('O')
      write('\n')
   write('\n')

proc randomWeights(ws: var Weights) =
   # start with random weights
   randomize() # seed random-number generator
   for i in 0 .. 2:
      ws[i] = rand(-1.0..1.0)

func feedForward(ins: Inputs; ws: Weights): int =
   # the perceptron outputs 1 if the sum of its inputs multiplied by
   # its input weights is positive, otherwise -1
   var sum = 0.0
   for i in 0 .. 2:
      sum = sum + ins[i].float * ws[i]
   result = if sum > 0.0: 1 else: -1

proc showOutput(ws: Weights) =
   var inputs: Inputs
   inputs[2] = 1 # bias
   for y in countdown(10, -9):
      for x in -9 .. 10:
         inputs[0] = x
         inputs[1] = y
         if feedForward(inputs, ws) == 1:
            write('#')
         else:
            write('O')
      write('\n')
   write('\n')

proc train(ws: var Weights; runs: int) =
   var inputs: Inputs
   inputs[2] = 1 # bias
   for i in 1 .. runs:
      for y in countdown(10, -9):
         for x in -9 .. 10:
            inputs[0] = x
            inputs[1] = y
            let error = (targetOutput(x, y) - feedForward(inputs, ws)).float
            for j in 0 .. 2:
               ws[j] = ws[j] + error * inputs[j].float * 0.01
               # 0.01 is the learning constant

proc main() =
   var weights: Weights
   echo("Target output for the function f(x) = 2x + 1:")
   showTargetOutput()
   randomWeights(weights)
   echo("Output from untrained perceptron:")
   showOutput(weights)
   train(weights, 1)
   echo("Output from perceptron after 1 training run:")
   showOutput(weights)
   train(weights, 4)
   echo("Output from perceptron after 5 training runs:")
   showOutput(weights)

main()
