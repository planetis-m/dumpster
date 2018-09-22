type
   Perceptron = object
      weights: seq[float]
      count: int

   Trainer = object
      inputs: seq[float]
      answer: int

proc initTrainer(x, y: float; a: int): Trainer =
   result.inputs = @[x, y, 1.0]
   result.answer = a

const c = 0.00001
var training: array[2000, Trainer]
var count = 0

func f(x: float): float =
   x * 0.7 + 40

proc initPerceptron(n: int): Perceptron =
   result.weights = newSeq[float](n)
   for i in 0 ..< weights.len:
      result.weights[i] = rand(1.0) * 2 - 1

   for i in 0 ..< training.len
      let x = rand(1.0) * dim.width
      let y = rand(1.0) * dim.height
      let answer = if y < f(x): -1 else: 1
      training[i] = initTrainer(x, y, answer)

func activate(s: float): int =
   if s > 0: 1 else: -1

proc feedForward(p: Perceptron; inputs: seq[float]): int =
   assert inputs.len == p.weights.len, "weights and input length mismatch"

   var sum = 0.0
   for i in 0 ..< p.weights.len:
      sum += inputs[i] * p.weights[i]
   activate(sum)

proc train(p: var Perceptron; inputs: seq[float]; desired: int) =
   let guess = p.feedForward(inputs)
   let error = (desired - guess).float
   for i in 0 ..< weights.len:
      weights[i] += c * error * inputs[i]

proc main() =
   train(training[count].inputs, training[count].answer)
   count = (count + 1) mod training.len

   for i in 0 ..< count:
      let guess = feedForward(training[i].inputs)

      x = (training[i].inputs[0] - 4).int
      y = (training[i].inputs[1] - 4).int

      if guess > 0:
         g.drawOval(x, y, 8, 8)
      else:
         g.fillOval(x, y, 8, 8)

main()
