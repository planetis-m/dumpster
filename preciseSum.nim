# From: https://bugs.python.org/file10357/msum4.py
import fenv, math

const
   maxExp = float.Exponent
   twoPow = pow(2.0, maxExp - 1.0)

template isNan(x): bool =
   classify(x) == fcNan

template isInf(x): bool =
   classify(x) in {fcNegInf, fcInf}

template isFinite(x): bool =
   classify(x) notin {fcNegInf, fcInf, fcNan}

func sameSign[T: SomeFloat](x, y: T): bool {.inline.} =
   x >= 0.0 == y >= 0.0

func twoSum[T: SomeFloat](x, y: T): (T, T) {.inline.} =
   # assumes that abs(x) >= abs(y)
   let hi = x + y
   let lo = y - (hi - x)
   result = (hi, lo)

func crsum[T: SomeFloat](partials: seq[T]): T =
   # Compute the sum of a list of nonoverlapping floats.
   #
   # On input, partials is a list of nonzero, nonspecial,
   # nonoverlapping floats, strictly increasing in magnitude, but
   # possibly not all having the same sign.
   #
   # On output, the sum of partials gives the error in the returned
   # result, which is correctly rounded (using the round-half-to-even
   # rule).  The elements of partials remain nonzero, nonspecial,
   # nonoverlapping, and increasing in magnitude.
   #
   # Assumes IEEE 754 float format and semantics.
   if partials.len == 0:
      return 0.0
   # sum from the top, stopping as soon as the sum is inexact.
   result = partials.pop()
   while partials.len > 0:
      var lo = T(0)
      (result, lo) = twoSum(result, partials.pop())
      if lo != 0:
         partials.add(lo)
         break
   # adjust for correct rounding if necessary
   if partials.len >= 2 and sameSign(partials[^1], partials[^2]) and
         result + 2 * partials[^1] - result == 2 * partials[^1]:
      result += 2 * partials[^1]
      partials[^1] = -partials[^1]

func msum*[T: SomeFloat](v: Iterable[T]): T =
   ## Full precision sum of values in iterable. Returns the value of the
   ## sum, rounded to the nearest representable floating-point number
   ## using the round-half-to-even rule
   # Stage 1: accumulate partials
   partials = @[0.0]
   for x in v:
      case classify(x)
      of fcNan:
         return x
      of fcNegInf, fcInf:
         partials[0] += x
      else:
         i = 1
         for y in partials[1:]:
               if abs(x) < abs(y):
                  swap(x, y)
               var (hi, lo) = twosum(x, y)
               if isInf(hi):
                  let sign = if hi > 0: 1 else: -1
                  var x = x - twopow * sign - twopow * sign
                  partials[0] += sign
                  if abs(x) < abs(y):
                     swap(x, y)
                  (hi, lo) = twoSum(x, y)
               if lo != 0:
                  partials[i] = lo
                  i += 1
               x = hi
         partials[i:] = [x] if x else []

   # special cases arising from infinities
   case classify(partials[0])
   of fcNegInf, fcInf:
      return partials[0]
   of fcNan:
      raise newException(ValueError, "infinities of both signs in summands")

   # Stage 2: sum partials[1:] + 2**exp_max * partials[0]
   if abs(partials[0]) == 1.0 and partials.len > 1 and
         not sameSign(partials[^1], partials[0]):
      # problem case: decide whether result is representable
      let (hi, lo) = twoSum(partials[0] * twoPow, partials[^1] / 2)
      if isInf(2 * hi):
         # overflow, except in edge case...
         if hi + 2 * lo - hi == 2 * lo and
               partials.len > 2 and sameSign(lo, partials[^2]):
            return 2*(hi + 2 * lo)
      else:
         partials[-1:] = [2*lo, 2*hi] if lo else [2*hi]
         partials[0] = 0.0

   if partials[0] == 0:
      return crsum(partials[1 .. ^1])
   raise newException(OverflowError, "overflow in msum")

when isMainModule:
   import strformat

   block test:
      let testValues = [
         (@[], 0.0),
         (@[0.0], 0.0),
         (@[1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50], 1e-100),
         (@[1e308, 1e308, -1e308], 1e308),
         (@[-1e308, 1e308, 1e308], 1e308),
         (@[1e308, -1e308, 1e308], 1e308),
         (@[pow(2.0, 1023), pow(2.0, 1023), pow(-2.0, 1000)], 1.7976930277114552e+308),
         (@[twopow, twopow, twopow, twopow, -twopow, -twopow, -twopow], 8.9884656743115795e+307),
         (@[2.0, 53, -0.5, pow(-2.0, -54)], pow(2.0, 53)-1.0),
         (@[2.0, 53, 1.0, pow(2.0, -100)], pow(2.0, 53)+2.0),
         (@[2.0, 53+10.0, 1.0, pow(2.0, -100)], pow(2.0, 53)+12.0),
         (@[2.0, 53-4.0, 0.5, pow(2.0, -54)], pow(2.0, 53-3.0)),
         (@[pow(2.0, 1023-pow(2.0,970)), -1.0, pow(2.0,1023)], 1.7976931348623157e+308),
         # (@[float_info.max, float_info.max*2.**-54], float_info.max),
         # (@[float_info.max, float_info.max*2.**-53], OverflowError),
         # (@[1./n for n in range(1, 1001)], 7.4854708605503451),
         # (@[(-1.)**n/n for n in range(1, 1001)], -0.69264743055982025),
         # (@[1.7**(i+1)-1.7**i for i in range(1000)] + [-1.7**1000], -1.0),
         (@[Inf, -Inf, Nan], Nan),
         (@[Nan, Inf, -Inf], Nan),
         (@[Inf, Nan, Inf], Nan),
         (@[Inf, Inf], Inf),
         # (@[Inf, -Inf], ValueError),
         (@[-Inf, 1e308, 1e308, -Inf], -Inf),
         # (@[2.0**1023-2.0**970, 0.0, 2.0**1023], OverflowError),
         # (@[2.0**1023-2.0**970, 1.0, 2.0**1023], OverflowError),
         # (@[2.0**1023, 2.0**1023], OverflowError),
         # (@[2.0**1023, 2.0**1023, -1.0], OverflowError),
         # (@[twopow, twopow, twopow, twopow, -twopow, -twopow], OverflowError),
         # (@[twopow, twopow, twopow, twopow, -twopow, twopow], OverflowError),
         # (@[-twopow, -twopow, -twopow, -twopow], OverflowError),
         # (@[2.**1023, 2.**1023, -2.**971], float_info.max),
         # (@[2.**1023, 2.**1023, -2.**970], OverflowError),
         # (@[-2.**970, 2.**1023, 2.**1023, -2.**-1074], float_info.max),
         # (@[2.**1023, 2.**1023, -2.**970, 2.**-1074], OverflowError),
         # (@[-2.**1023, 2.**971, -2.**1023], -float_info.max),
         # (@[-2.**1023, -2.**1023, 2.**970], OverflowError),
         # (@[-2.**1023, -2.**1023, 2.**970, 2.**-1074], -float_info.max),
         # (@[-2.**-1074, -2.**1023, -2.**1023, 2.**970], OverflowError),
         (@[pow(2.0, 930), pow(-2.0, 980), pow(2.0, 1023), pow(2.0, 1023), twopow, -twopow], 1.7976931348622137e+308),
         (@[pow(2.0, 1023), pow(2.0, 1023), -1e307], 1.6976931348623159e+308),
         (@[1e16, 1.0, 1e-16], 10000000000000002.0)
      ]
      var i = 0
      for vals, s in testValues.items:
         let m = msum(vals)
         assert(m == s or isNan(m) and isNan(s), &"Test {i} failed: got {m}, expected {s} for msum({vals}).")
         i.inc
