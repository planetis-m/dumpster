import math, random, color

const
   epoch = 1500
   training = 1
   transition = 2
   show = 3

var perceptron: Perceptron
var counter = 0
var learnRate = 0.02
var state = training

proc setup() =
   createCanvas(800, 600)
   clearBack()
   perceptron = newPerceptron(2)

proc draw() =
   case state:
   of training: training()
   of transition: transition()
   of show: show()

proc clearBack() =
   background(0)
   stroke(255)
   strokeWeight(4)

   let x = width
   line(0, 0, x, lineDef(x))
 
proc transition() =
   clearBack()
   state = show

func lineDef(x) =
   0.75 * x

proc training() =
   var a = rand(width)
   var b = rand(height)

   let lDef = if lineDef(a) > b: -1 else: 1
 
   perceptron.setInput([a, b])
   perceptron.feedForward()
   var pRes = perceptron.getOutput()
   var match = (pRes == lDef)
   var clr: Color

   if not match:
      let err = (pRes - lDef) * learnRate
      perceptron.adjustWeights(err)
 
      clr = color(255, 0, 0)
 
   else:
      clr = color(0, 255, 0)

   noStroke()
   fill(clr)
   ellipse(a, b, 4, 4)
 
   counter.inc
   if counter == epoch:
      state = transition

proc show() =
   var
      a = rand(width)
      b = rand(height)
      clr: Color

   perceptron.setInput([a, b])
   perceptron.feedForward()
   let pRes = perceptron.getOutput()

   if pRes < 0:
      clr = color(255, 0, 0)
   else:
      clr = color(0, 255, 0)

   noStroke()
   fill(clr)
   ellipse(a, b, 4, 4)

type
   Perceptron = ref object
      inputs, weights: seq[float]
      output, bias: int

proc newPerceptron(inNumber: int): Perceptron =
   this.inputs = @[]
   this.weights = @[]
   this.output = 0
   this.bias = 1

   # one more weight for bias
   for i in 0 .. inNumber:
      this.weights.add(rand(1.0))

proc activation(a: float): int =
   if tanh(a) < 0.5: 1 else: -1

proc feedForward(this: Perceptron) =
   var sum = 0.0
   for i in 0 ..< this.inputs.len:
      sum += this.inputs[i] * this.weights[i]

   sum += this.bias * this.weights[^1]
   this.output = activation(sum)

proc getOutput(this: Perceptron) =
   this.output

proc setInput(this: Perceptron; inputs: seq[float]) =
   this.inputs = []
   for i in 0 ..< inputs.len:
      this.inputs.add(inputs[i])

proc adjustWeights(this: Perceptron; err: float) =
   for i in 0 ..< this.weights.len:
      this.weights[i] += err * this.inputs[i]
