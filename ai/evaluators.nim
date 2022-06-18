import math

type
  Evaluator* = concept
    proc evaluate(self: Self, value: float32): float32

type
  LinearEvaluator* = object
    xa, ya: float32
    dyOverDx: float32

proc initLinearEvaluator*(xa, ya, xb, yb: float32): LinearEvaluator =
  # defualt 0.0, 0.0, 100.0, 100.0
  result = LinearEvaluator(xa: xa, ya: ya, dyOverDx: (yb - ya) / (xb - xa))

proc evaluate*(self: LinearEvaluator, value: float32): float32 =
  result = clamp(self.ya + self.dyOverDx * (value - self.xa), 0, 1)

type
  PowerEvaluator* = object
    xa, ya, xb: float32
    power: float32
    dy: float32

proc initPowerEvaluator*(power, xa, ya, xb, yb: float32): PowerEvaluator =
  # 2.0, 0.0, 0.0, 100.0, 100.0
  result = PowerEvaluator(
    power: clamp(power, 0, 10000),
    dy: yb - ya,
    xa: xa, ya: ya, xb: xb
  )

proc evaluate*(self: PowerEvaluator, value: float32): float32 =
  let cx = clamp(value, self.xa, self.xb)
  result = self.dy * ((cx - self.xa) / (self.xb - self.xa)).pow(self.power) + self.ya

type
  SigmoidEvaluator* = object
    xa, xb: float32
    k: float32
    twoOverDx: float32
    xMean, yMean: float32
    dyOverTwo: float32
    oneMinusK: float32

proc initSigmoidEvaluator*(k, xa, ya, xb, yb: float32): SigmoidEvaluator =
  # -0.5, 0.0, 0.0, 100.0, 100.0
  let k = clamp(k, -0.99999'f32, 0.99999'f32)
  result = SigmoidEvaluator(
    xa: xa,
    xb: xb,
    twoOverDx: abs(2 / (xb - ya)),
    xMean: (xa + xb) / 2,
    yMean: (ya + yb) / 2,
    dyOverTwo: (yb - ya) / 2,
    oneMinusK: 1 - k,
    k: k
  )

proc evaluate*(self: SigmoidEvaluator, x: float32): float32 =
  let cxMinusXMean = clamp(x, self.xa, self.xb) - self.xMean
  let numerator = self.twoOverDx * cxMinusXMean * self.oneMinusK
  let denominator = self.k * (1 - 2 * abs(self.twoOverDx * cxMinusXMean)) + 1
  result = self.dyOverTwo * (numerator / denominator) + self.yMean
