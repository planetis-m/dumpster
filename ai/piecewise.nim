import vmath

const
  MaxControlPoints = 4

type
  PiecewiseLinearCurve* = object
    controlPoints: array[MaxControlPoints, Point2]

proc initPiecewise*(controlPoints: array[MaxControlPoints, Point2]): PiecewiseLinearCurve =
  assert controlPoints[0].x == 0, "first data point should be at zero"
  result = PiecewiseLinearCurve(controlPoints: controlPoints)

proc maxT(p: PiecewiseLinearCurve): float32 =
  result = p.controlPoints[0].x
  for i in 1 ..< p.controlPoints.len:
    if result < p.controlPoints[i].x: result = p.controlPoints[i].x

proc eval(self: PiecewiseLinearCurve; t: float32): float32 =
  assert t <= self.maxT, "t is too big to be defined by this PiecewiseLinearCurve"
  assert t > 0, "negative t is not defined by this PiecewiseLinearCurve"
  var idx = 1
  while idx < self.controlPoints.high and self.controlPoints[idx].x < t:
    inc idx
  let cp = self.controlPoints[idx]
  let last = self.controlPoints[idx - 1]
  # linear interpolation
  result = (last.y * (cp.x - t) + cp.y * (t - last.x)) / (cp.x - last.x)

let x = initPiecewise([point2(0, 0), point2(1, 1), point2(3, 1), point2(4, 2)])
echo eval(x, 2)
