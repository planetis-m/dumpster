import math

type
   Averager* = object
      count: int
      x: float

proc feed*(self: var Averager; value: float) =
   self.x += value
   inc(self.count)

proc done*(self: var Averager): float =
   result = self.x / self.count.float
   self.count = 0
   self.x = 0

type
   Deviator* = object
      count: int
      x: float

proc feed*(self: var Deviator; value, mean: float) =
   let val = (value - mean)
   self.x += val * val
   inc(self.count)

proc done*(self: var Deviator; population: bool = true): float =
   if population:
      result = self.x / self.count.float
   else:
      result = self.x / (self.count - 1).float
   self.count = 0
   self.x = 0

type
   Correlator* = object
      ## Computes Pearson R from a stream of x/y values.
      xy, x2, y2: float

proc feed*(self: var Correlator; x, y: float) =
   self.xy += x * y
   self.x2 += x * x
   self.y2 += y * y

proc done*(self: var Correlator): float =
   result = self.xy / sqrt(self.x2 * self.y2)
   self.xy = 0
   self.x2 = 0
   self.y2 = 0

proc linear_regression*(r, mx, my, sx, sy: float; slope, intercept: out float) =
   ## Computes the slope and intercept for a linear regression line given
   ## correlation R, means of x and y, and standard deviations of x and y.
   slope = r * sy / sx
   intercept = my - slope * mx
