import vmath

type
  PiecewiseLinearCurve* = object
    data: seq[Point2]

proc initPlc*(data: seq[Point2]): PiecewiseLinearCurve =
  assert(data[0].x == 0, "first data point should be at zero")
  result = PiecewiseLinearCurve(data: data)

proc maxT(p: PiecewiseLinearCurve): float32 =
  result = p.data[0].x
  for i in 1 ..< p.data.len:
    if result < p.data[i].x: result = p.data[i].x

proc evaluate(p: PiecewiseLinearCurve; t: float32): float32 =
  assert(t <= p.maxT, "t is too big to be defined by this PiecewiseLinearCurve")
  assert(t > 0, "negative t is not defined by this PiecewiseLinearCurve")
  var
    p0: Point2
    p1: Point2
  for point in p.data.items:
    if point.x > t:
      p1 = point
      break
    else:
      p0 = point
  # linear interpolation
  result = (p0.y * (p1.x - t) + p1.y * (t - p0.x)) / (p1.x - p0.x)

let x = initPlc(@[point2(0, 0), point2(1, 1), point2(3, 1), point2(4, 2)])
echo x.evaluate(2)
