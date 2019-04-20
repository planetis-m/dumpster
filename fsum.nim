# math.fsum relies on exact rounding for correct operation.
# There's a known problem with IA32 floating-point that causes
# inexact rounding in some situations, and will cause the
# math.fsum tests below to fail; see issue #2937.  On non IEEE
# 754 platforms, and on IEEE 754 platforms that exhibit the
# problem described in issue #2937, we simply skip the whole
# test.

# Python version of math.fsum, for comparison.  Uses a
# different algorithm based on frexp, ldexp and integer
# arithmetic.
import fenv, math

const
   mantDig = float.mantissaDigits
   etiny = float.minExponent - mantDig

type
   Iterable[T] = concept c
      for x in items(c): x is T

proc bitLen(num: int): int =
   # returns the number of bits necessary to represent an integer in binary
   # excluding the sign and leading zeros.
   var num = num
   while num > 0:
      result.inc
      num = num shr 1

func fsum*[T: SomeFloat](v: Iterable[T]): T =
   # Full precision summation. Compute sum(v) without any intermediate
   # accumulation of error. Based on the 'lsum' function at
   # http://code.activestate.com/recipes/393090/
   var
      tmant = 0
      texp = 0
   for x in v:
      var exp = 0
      let mantf = frexp(x, exp)
      var mant = int(ldexp(mantf, mantDig))
      exp = exp - mantDig
      if texp > exp:
         tmant = tmant shl (texp - exp)
         texp = exp
      else:
         mant = mant shl (exp - texp)
      tmant += mant
   # Round tmant * 2**texp to a float.
   let tail = max(bitLen(abs(tmant)) - mantDig, etiny - texp)
   if tail > 0:
      let h = 1 shl (tail - 1)
      tmant = tmant mod (2 * h) + int(bool((tmant and h) and (tmant and 3 * h - 1)))
      texp += tail
   result = ldexp(tmant, texp)

when isMainModule:
   import strformat, strutils

   {.floatChecks: on.}
   let testValues = [
      (@[], 0.0),
      (@[0.0], 0.0),
      (@[1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50], 1e-100),
      (@[pow(2.0, 53), -0.5, pow(-2.0, -54)], pow(2.0, 53)-1.0),
      (@[pow(2.0, 53), 1.0, pow(2.0, -100)], pow(2.0, 53)+2.0),
      (@[pow(2.0, 53+10.0), 1.0, pow(2.0, -100)], pow(2.0,53)+12.0),
      (@[pow(2.0, 53-4.0), 0.5, pow(2.0, -54)], pow(2.0, 53)-3.0),
      # (@[1.0/n for n in range(1, 1001)],
      #    float.fromhex("0x1.df11f45f4e61ap+2")),
      # (@[(-1.0)**n/n for n in range(1, 1001)],
      #    parseHexFloat("-0x1.62a2af1bd3624p-1")),
      # (@[pow(1.7, i+1)-pow(1.7, i) for i in range(1000)] + [pow(-1.7, 1000)], -1.0),
      (@[1e16, 1.0, 1e-16], 10000000000000002.0),
      (@[1e16 - 2.0, 1.0 - pow(2.0,-53), -(1e16-2.0), -(1.0-pow(2.0,-53))], 0.0),
      # # exercise code for resizing partials array
      # (@[pow(2.0, n) - pow(2.0, (n + 50)) + pow(2.0, (n+52)) for n in range(-1074, 972, 2)] +
      #    [pow(-2.0, 1022)],
      #    parseHexFloat("0x1.5555555555555p+970")),
   ]

   var i = 0
   for vals, expected in items(testValues):
      try:
         actual = fsum(vals)
      except OverflowError:
         assert false,
            &"test {i} failed: got OverflowError, expected {expected} for fsum({vals})"
      except ValueError:
         assert false,
            &"test {i} failed: got ValueError, expected {expected} for fsum({vals})"
      assert actual == expected
      i.inc

   from random import rand, gauss, shuffle
   import sequtils

   for j in 1 .. 1000:
      var vals = repeat(@[7, 1e100, -7, -1e100, -9e-20, 8e-20], 10)
      var s = 0.0
      for i in 1 .. 200:
         let v = pow(gauss(0, rand(1.0)), 7) - s
         s += v
         vals.add(v)
      shuffle(vals)

      assert msum(vals) == fsum(vals)
