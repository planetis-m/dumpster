# 'borrowed' from fowlmouth at https://github.com/fowlmouth/nimlibs
# available -define:
#   NeuralFloat32 - makes float type used be float32
import random, strutils, json, math

randomize()

when defined(NeuralFloat32):
   type NeuralFloat* = float32
else:
   type NeuralFloat* = float64

type
   ActivationFunction* = object
      fn*, deriv*: proc(x: NeuralFloat): NeuralFloat {.nimcall.}

   NeuralNet* = object
      layerSizes*: seq[int]
      outputs: seq[seq[NeuralFloat]] # outputs
      deltas: seq[seq[NeuralFloat]] # error values
      weights, previousWeights: seq[seq[seq[NeuralFloat]]]
      learningRate, momentum: NeuralFloat
      activationf: ActivationFunction

   TrainingData* = object
      inputs, target: seq[NeuralFloat]

let
   identity* = ActivationFunction(
      fn: func(x: NeuralFloat): NeuralFloat = x,
      deriv: func(fx: NeuralFloat): NeuralFloat = 1)

   sigmoid* = ActivationFunction(
      fn: func(x: NeuralFloat): NeuralFloat = 1 / (1 + exp(-x)),
      deriv: func(fx: NeuralFloat): NeuralFloat = fx * (1 - fx))

   tanh* = ActivationFunction(
      fn: func(x: NeuralFloat): NeuralFloat = tanh(x),
      deriv: func(fx: NeuralFloat): NeuralFloat = (let t = tanh(fx); 1 - t * t))

   relu* = ActivationFunction(
      fn: func(x: NeuralFloat): NeuralFloat = max(0, x),
      deriv: func(fx: NeuralFloat): NeuralFloat = (if fx == 0: 0 else: 1))

   leakyrelu* = ActivationFunction(
      fn: func(x: NeuralFloat): NeuralFloat = (if x < 0: 0.01 * x else: x),
      deriv: func(fx: NeuralFloat): NeuralFloat = (if fx < 0: 0.01 else: 1))

proc activationFunc*(n: NeuralNet): ActivationFunction {.inline.} =
   n.activationf

proc `activationFunc=`*(n: var NeuralNet, f: ActivationFunction) {.inline.} =
   n.activationf = f

proc numLayers*(n: NeuralNet): int {.inline.} =
   n.layerSizes.len

proc numInputs*(n: NeuralNet): int {.inline.} =
   n.layerSizes[0]

proc numOutputs*(n: NeuralNet): int {.inline.} =
   n.layerSizes[n.layerSizes.len - 1]

proc getOutputs*(n: NeuralNet): seq[NeuralFloat] {.inline.} =
   n.outputs[n.numLayers - 1]

proc getOutput*(n: NeuralNet; i: int): NeuralFloat {.inline.} =
   n.outputs[n.numLayers - 1][i]

template copy*(n: NeuralNet): NeuralNet =
   NeuralNet(
      layerSizes: n.layerSizes,
      outputs: n.outputs,
      weights: n.weights,
      activationf: n.activationf)

proc initNeuralNet*(layers: seq[int]): NeuralNet =
   result.layerSizes = layers
   result.activationf = sigmoid

   let layerCount = result.numLayers
   newSeq(result.outputs, layerCount)
   newSeq(result.weights, layerCount)

   for i, len in layers:
      newSeq(result.outputs[i], len)
      if i > 0:
         newSeq(result.weights[i], len)
         for j in 0 ..< len:
            newSeq(result.weights[i][j], layers[i - 1] + 1)

proc prepareForTraining*(n: var NeuralNet; learningRate, momentum: NeuralFloat) =
   n.learningRate = learningRate
   n.momentum = momentum

   # initialize layers
   let layerCount = n.numLayers
   newSeq(n.deltas, layerCount)
   newSeq(n.previousWeights, layerCount)

   for i in 1 ..< layerCount:
      let len = n.layerSizes[i]
      newSeq(n.deltas[i], len)
      newSeq(n.previousWeights[i], len)
      for j in 0 ..< len:
         newSeq(n.previousWeights[i][j], n.layersizes[i - 1] + 1)
         for k in 0 ..< n.layersizes[i - 1] + 1:
            n.weights[i][j][k] = rand(1.0)

proc feed*(n: var NeuralNet; input: seq[NeuralFloat]) =
   # assign inputs
   for i in 0 ..< n.numInputs:
      n.outputs[0][i] = input[i]

   for i in 1 ..< n.numLayers:
      for j in 0 ..< n.layerSizes[i]:
         var sum = 0.0
         for k in 0 ..< n.layerSizes[i - 1]:
            sum += n.outputs[i - 1][k] * n.weights[i][j][k]
         sum += n.weights[i][j][n.layerSizes[i - 1]]
         n.outputs[i][j] = n.activationf.fn(sum)

proc backProp*(n: var NeuralNet; input, target: seq[NeuralFloat]) =
   assert target.len == n.numOutputs
   assert input.len == n.numInputs

   # update output values
   n.feed input

   # find deltas for output layer
   let
      lastLayer = n.numLayers - 1
      numLayers = n.numLayers

   for i in 0 ..< n.numOutputs:
      n.deltas[lastLayer][i] =
         n.activationf.deriv(n.outputs[lastLayer][i]) *
         (target[i] - n.outputs[lastLayer][i])

   # find deltas for hidden layer
   for i in countdown(numLayers - 2, 1):
      for j in 0 ..< n.layerSizes[i]:
         var sum = 0.0
         for k in 0 ..< n.layerSizes[i+1]:
            sum += n.deltas[i+1][k] * n.weights[i+1][k][j]
         n.deltas[i][j] =
            n.activationf.deriv(n.outputs[i][j]) * sum
         #  n.outputs[i][j] * (1.0 - n.outputs[i][j]) * sum

   # apply momentum
   if n.momentum != 0.0:
      for i in 1 ..< numLayers:
         for j in 0 ..< n.layerSizes[i]:
            for k in 0 ..< n.layerSizes[i-1]:
               n.weights[i][j][k] +=
                  n.momentum * n.previousWeights[i][j][k]
            n.weights[i][j][n.layerSizes[i-1]] +=
               n.momentum * n.previousWeights[i][j][n.layerSizes[i-1]]

   for i in 1 ..< numLayers:
      for j in 0 ..< n.layerSizes[i]:
         for k in 0 ..< n.layerSizes[i-1]:
            n.previousWeights[i][j][k] =
               n.learningRate * n.deltas[i][j] * n.outputs[i-1][k]
            n.weights[i][j][k] +=
               n.previousWeights[i][j][k]
         n.previousWeights[i][j][n.layerSizes[i-1]] =
            n.learningRate * n.deltas[i][j]
         n.weights[i][j][n.layerSizes[i-1]] +=
            n.previousWeights[i][j][n.layerSizes[i-1]]

proc meanSquareError(n: NeuralNet; target: seq[NeuralFloat]): NeuralFloat =
   for i in 0 ..< n.numOutputs:
      result +=
         (target[i] - n.getOutput(i)) * (target[i] - n.getOutput(i))
   result = result / 2.0

proc ff(f: NeuralFloat; prec = 2): string = formatFloat(f, ffDecimal, prec)

proc train*(n: var NeuralNet; data: seq[TrainingData];
            numIters = 1_000_000; threshold = 0.000001) =
   for iter in 0 ..< numIters:
      var correct = 0
      when defined(DebugNeural):
         var avg_mse = 0.0

      for i in 0 ..< data.len:
         n.backProp data[i].inputs, data[i].target

         let mse = n.meanSquareError(data[i].target)
         if mse < threshold:
            correct.inc
         when defined(DebugNeural):
            avg_mse += mse

      if correct == data.len:
         when defined(DebugNeural):
            echo "Network trained in ", iter + 1, " iterations."
         break

      when defined(DebugNeural):
         if iter mod int(numIters / 10) == 0:
            avg_mse = avg_mse / data.len.NeuralFloat
            echo "MSE: ", ff(avg_mse, 8)

proc predict*(n: var NeuralNet; input: seq[NeuralFloat]): NeuralFloat =
   n.feed input
   n.getOutput(0)

proc toFloat*(j: JsonNode): NeuralFloat =
   case j.kind
   of JInt:
      return j.num.NeuralFloat
   of JFloat:
      return j.fnum.NeuralFloat
   of JString:
      return j.str.parseFloat
   else:
      discard

proc getFloat(j: JsonNode; field: string; default = 0.0): NeuralFloat =
   if j.hasKey(field):
      j[field].toFloat
   else:
      default

proc toInt*(j: JsonNode): int =
   case j.kind
   of JInt:
      return j.num.int
   of JFloat:
      return j.fnum.int
   of JString:
      return j.str.parseInt
   else:
      discard

proc getInt(j: JsonNode; field: string; default = 0): int =
   if j.hasKey(field):
      j[field].toInt
   else:
      default

proc setActivationFunc*(n: var NeuralNet; function: string) =
   case function.toLowerAscii
   of "tanh":
      n.activationf = tanh
   of "logistic", "sigmoid":
      n.activationf = sigmoid
   else:
      raise newException(ValueError, "activation function not recognized: " & function)

proc loadNeuralNet*(data: JsonNode): NeuralNet =
   ## loads a neural net defined in JSON. see the bottom of the page for an example
   ## if there is a "training" section the net will be trained before being returned.
   var layers: seq[int] = @[]
   for n in data["layers"]:
      layers.add n.toInt

   result = initNeuralNet(layers)

   if data.hasKey("activation_function"):
      result.setActivationFunc data["activation_function"].str

   if data.hasKey("weights"):
      for layer in 1 ..< result.numLayers:
         for i in 0 ..< data["weights"][layer - 1].len:
            for j in 0 ..< data["weights"][layer - 1][i].len:
               result.weights[layer][i][j] = data["weights"][layer - 1][i][j].toFloat

   elif data.hasKey("training"):
      var trainingData: seq[TrainingData] = @[]

      let
         training = data["training"]
         iterations = training.getInt("iterations", 500_000)
         threshold = training.getFloat("threshold", 0.00001)
         learningRate = training.getFloat("learning-rate", 0.3)
         momentum = training.getFloat("momentum", 0.1)

      for t in training["set"]:
         var data: TrainingData

         data.inputs = newSeq[NeuralFloat](t[0].len)
         data.target = newSeq[NeuralFloat](t[1].len)

         for i, v in t[0].elems:
            data.inputs[i] = v.toFloat
         for i, v in t[1].elems:
            data.target[i] = v.toFloat

         trainingData.add data

      result.prepareForTraining(learningRate, momentum)
      result.train trainingData, iterations, threshold

proc loadNeuralNet*(file: string): NeuralNet =
   loadNeuralNet(json.parseFile(file))

proc `%`*(n: NeuralNet): JsonNode =
   let layers = newJarray()
   for len in n.layerSizes:
      layers.add(%len)

   let weights = newJarray()
   for i in 1 ..< n.numLayers:
      let x = newJarray()
      for j in n.weights[i]:
         let y = newJarray()
         for k in j:
            y.add(%k)
         x.add y
      weights.add x

   %{"layers": layers, "weights": weights}

proc save*(n: NeuralNet; file: string) =
   let j = %n
   writeFile(file, j.pretty)

when isMainModule:
   import os, times

   let trainingData = {
      "or": %*{
         "layers": [2, 1],
         "training": {
            "set": [
               [[0, 1], [1]],
               [[1, 0], [1]],
               [[0, 1], [1]],
               [[0, 0], [0]]]
            }
         },
      "xor": %*{
         "layers": [2, 2, 1],
         "training": {
            "set": [
               [[0, 1], [1]],
               [[1, 0], [1]],
               [[1, 1], [0]],
               [[0, 0], [0]]]
            }
         }
      }

   for name, data in trainingData.items:
      echo "Training '", name, "'"
      let start = epochTime()
      var net = loadNeuralNet(data)
      echo "finished in ", ff(epochTime() - start, 4), " seconds"

      var inputs = newSeq[NeuralFloat](net.numInputs)
      for s in data["training"]["set"]:
         for i, v in s[0].elems:
            inputs[i] = v.toFloat
         echo "  ", inputs, " = ", net.predict(inputs)
